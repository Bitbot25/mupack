#![feature(const_mut_refs, const_trait_impl)]

use std::fmt;

#[derive(Debug)]
pub enum MuError {
    Io(std::io::Error),
    StarveMin(usize),
}

pub type DecResult<T> = std::result::Result<T, MuError>;

#[const_trait]
pub trait MuStarve {
    fn min_starve() -> usize;
}

pub trait MufieldT: Sized + ~const MuStarve {
    fn enc<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()>;
    fn dec<R: std::io::Read>(r: &mut R) -> DecResult<Self>;
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub struct Hello {
    pub byte: u8,
    pub maybe_exists: Option<u8>,
}

impl MufieldT for u8 {
    #[tracing::instrument(skip(r))]
    fn dec<R: std::io::Read>(r: &mut R) -> DecResult<Self> {
        let mut b = [0; 1];

        match r.read_exact(&mut b) {
            Ok(_) => Ok(b[0]),
            Err(ioerr) => {
                if let std::io::ErrorKind::WouldBlock = ioerr.kind() {
                    Err(MuError::StarveMin(Self::min_starve()))
                } else {
                    tracing::warn!(ioerr = ?&ioerr, "unknown IO error");
                    Err(MuError::Io(ioerr))
                }
            }
        }
    }

    fn enc<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        w.write_all(&[*self])
    }
}

impl const MuStarve for u8 {
    fn min_starve() -> usize {
        1
    }
}

impl PrimitiveExpand for u8 {
    fn expand(&self, out: &mut Vec<Primitive>) {
        out.push(Primitive::Int(IntPrimitive {
            val: *self as u64,
            max: 256,
        }));
    }
}

impl MufieldT for bool {
    fn enc<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        let b = *self as u8;
        u8::enc(&b, w)
    }

    fn dec<R: std::io::Read>(r: &mut R) -> DecResult<Self> {
        let b = u8::dec(r)?;
        Ok(b != 0)
    }
}

impl const MuStarve for bool {
    fn min_starve() -> usize {
        1
    }
}

impl PrimitiveExpand for bool {
    fn expand(&self, out: &mut Vec<Primitive>) {
        out.push(Primitive::Int(IntPrimitive {
            val: (*self as u8) as u64,
            max: 2,
        }));
    }
}

impl<T: MufieldT> MufieldT for Option<T> {
    fn enc<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        match self {
            Some(x) => {
                bool::enc(&true, w)?;
                T::enc(x, w)
            }
            None => bool::enc(&false, w),
        }
    }

    fn dec<R: std::io::Read>(r: &mut R) -> DecResult<Self> {
        let flag = starvepromote(bool::dec(r), T::min_starve())?;
        if flag {
            let x = T::dec(r)?;
            Ok(Some(x))
        } else {
            Ok(None)
        }
    }
}

impl<T: MufieldT + ~const MuStarve> const MuStarve for Option<T> {
    fn min_starve() -> usize {
        u8::min_starve() + T::min_starve()
    }
}

impl<T: PrimitiveExpand> PrimitiveExpand for Option<T> {
    fn expand(&self, out: &mut Vec<Primitive>) {
        match self {
            Some(x) => {
                bool::expand(&true, out);
                T::expand(x, out);
            }
            None => {
                bool::expand(&false, out);
            }
        }
    }
}

impl MufieldT for Hello {
    fn enc<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        u8::enc(&self.byte, w)?;
        Option::<u8>::enc(&self.maybe_exists, w)
    }

    fn dec<R: std::io::Read>(r: &mut R) -> DecResult<Self> {
        let byte = u8::dec(r)?;
        let maybe_exists = Option::<u8>::dec(r)?;
        Ok(Self { byte, maybe_exists })
    }
}

impl const MuStarve for Hello {
    fn min_starve() -> usize {
        u8::min_starve() + Option::<u8>::min_starve()
    }
}

impl PrimitiveExpand for Hello {
    fn expand(&self, out: &mut Vec<Primitive>) {
        self.byte.expand(out);
        self.maybe_exists.expand(out);
    }
}

fn starvepromote<T>(res: DecResult<T>, extra: usize) -> DecResult<T> {
    match res {
        DecResult::Ok(r) => DecResult::Ok(r),
        DecResult::Err(MuError::StarveMin(l)) => DecResult::Err(MuError::StarveMin(l + extra)),
        DecResult::Err(MuError::Io(ioerr)) => {
            // Should be converted to StarveMin
            assert_ne!(ioerr.kind(), std::io::ErrorKind::WouldBlock);
            DecResult::Err(MuError::Io(ioerr))
        }
    }
}

struct WaitBuffer<R: std::io::Read> {
    r: R,
    buf: Vec<u8>,
    bpos: usize,
}

impl<R: std::io::Read> WaitBuffer<R> {
    pub fn new(r: R) -> Self {
        WaitBuffer {
            r,
            buf: Vec::new(),
            bpos: 0,
        }
    }
}

// TODO: Wait in WaitBuffer::read instead
impl<R: std::io::Read> WaitBuffer<R> {
    #[tracing::instrument(skip(self))]
    pub fn ready(&mut self, n: usize) -> std::io::Result<()> {
        let mut rbuflen = self.buf.len();
        let buflen = rbuflen + n;

        self.buf.reserve(n);
        unsafe {
            self.buf.set_len(buflen);
        }

        tracing::trace!(n, "reserved {n} elements in buf");

        // FIXME: Make sure that a panic will not leave `self.buf` invalid by using a drop guard.
        // Callers may catch the panic and call the read function on error, which will cause
        // undefined behaviour.
        while rbuflen != buflen {
            let n = match self.r.read(&mut self.buf[rbuflen..]) {
                Ok(n) => n,
                Err(e) => {
                    // Remove uninitialized elements on error
                    tracing::trace!(
                        rbuflen,
                        "reset buf len to only contain initialized elements"
                    );
                    unsafe {
                        self.buf.set_len(rbuflen);
                    }
                    return Err(e);
                }
            };
            rbuflen += n;
        }
        Ok(())
    }

    fn read_populate(&mut self) -> std::io::Result<()> {
        let mut arr = [0; 128];
        let n = self.r.read(&mut arr)?;
        self.buf.extend_from_slice(&arr[..n]);

        Ok(())
    }
}

impl<R: std::io::Read> std::io::Read for WaitBuffer<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        // There are no new pre-read elements
        if self.buf.len() == self.bpos {
            self.read_populate()?;
        }

        let ibuflen = self.buf.len();
        let ncopy = (ibuflen - self.bpos).min(buf.len());

        let srcregion = &self.buf[self.bpos..self.bpos + ncopy];
        let destregion = &mut buf[..ncopy];

        destregion.copy_from_slice(srcregion);

        self.bpos += ncopy;

        // TODO: Flush buffer if all elements are read (and it has a very large capacity).

        Ok(ncopy)
    }
}

/// Abbreviaties a repeating array and removes elements beyond max_elements
struct ArrayPrettyPrint<'a, T: PartialEq + fmt::Debug> {
    arr: &'a [T],
    cutoff: Option<usize>,
    newline_threshold: Option<usize>,
}

impl<'a, T: PartialEq + fmt::Debug> ArrayPrettyPrint<'a, T> {
    pub fn new(arr: &'a [T], cutoff: Option<usize>, newline_threshold: Option<usize>) -> Self {
        Self {
            arr,
            cutoff,
            newline_threshold,
        }
    }
}

impl<'a, T: PartialEq + fmt::Debug> fmt::Debug for ArrayPrettyPrint<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn is_repeating<T: PartialEq>(arr: &[T], min_repeating: usize) -> Option<&T> {
            if arr.is_empty() {
                return None;
            }

            let first = &arr[0];
            let mut repeat = 0;
            for elem in arr.iter().skip(1) {
                if elem != first {
                    return None;
                }
                repeat += 1;
            }
            if repeat >= min_repeating {
                Some(first)
            } else {
                None
            }
        }

        if let Some(elem) = is_repeating(&self.arr, 1) {
            write!(f, "[{elem:?}; {0}]", self.arr.len())
        } else {
            let print_newlines =
                self.arr.len() > self.newline_threshold.unwrap_or(usize::MAX) && f.alternate();

            write!(f, "[")?;
            if let Some(last) = self.arr.last() {
                for (i, e) in self.arr.iter().take(self.arr.len() - 1).enumerate() {
                    let shear = i >= self.cutoff.unwrap_or(usize::MAX);
                    let postfix = match (shear, print_newlines) {
                        (true, true) => "\n...",
                        (true, false) => " ...",

                        (false, true) => "\n",
                        (false, false) => " ",
                    };
                    write!(f, "{:?},{postfix}", e)?;
                }

                let shear_last = self.arr.len() - 1 > self.cutoff.unwrap_or(usize::MAX);
                if shear_last {
                    write!(f, "...")?;
                } else {
                    write!(f, "{last:?}")?;
                }
            }
            write!(f, "]")
        }
    }
}

pub mod nonblock {
    use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt, ReadBuf};

    use crate::{MuError, MuStarve, MufieldT};
    use pin_project::pin_project;
    use std::{
        future::Future, io::Cursor, marker::PhantomData, mem::MaybeUninit, pin::Pin, task::Poll,
    };

    #[pin_project]
    pub struct UnpackFuture<T, R, const BUFSZ: usize = 256> {
        r: R,
        req: usize,
        _t: PhantomData<T>,
        _bufsz: PhantomData<[(); BUFSZ]>,
    }

    impl<const BUFSZ: usize, T: MufieldT + ~const MuStarve, R: AsyncRead + Unpin> Future
        for UnpackFuture<T, R, BUFSZ>
    {
        type Output = tokio::io::Result<T>;

        #[tracing::instrument(skip(self, cx))]
        fn poll(
            mut self: Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            let mut src_buf: [MaybeUninit<u8>; BUFSZ] =
                unsafe { MaybeUninit::uninit().assume_init() };
            let mut read_buf = ReadBuf::uninit(&mut src_buf);
            let bufrem = read_buf.remaining();

            match AsyncRead::poll_read(Pin::new(&mut self.r), cx, &mut read_buf) {
                Poll::Ready(Ok(_)) => {
                    if read_buf.remaining() == bufrem {
                        return Poll::Ready(Err(tokio::io::Error::from(
                            tokio::io::ErrorKind::UnexpectedEof,
                        )));
                    }
                }
                Poll::Ready(Err(e)) => {
                    return Poll::Ready(Err(e));
                }
                Poll::Pending => {
                    return Poll::Pending;
                }
            };

            let slice = read_buf.filled();

            if slice.len() >= self.req {
                tracing::trace!(
                    req = self.req,
                    has = slice.len(),
                    typename = std::any::type_name::<T>(),
                    "attempting parse"
                );
                let decoded = match T::dec(&mut Cursor::new(slice)) {
                    Ok(r) => r,
                    Err(MuError::StarveMin(min)) => {
                        self.req = min;
                        return Poll::Pending;
                    }
                    Err(MuError::Io(ioerr)) => return Poll::Ready(Err(ioerr)),
                };
                tracing::info!(typename = std::any::type_name::<T>(), "parse succeeded");
                Poll::Ready(Ok(decoded))
            } else {
                Poll::Pending
            }
        }
    }

    pub const fn unpack<'a, T: MufieldT + ~const MuStarve + 'a, R: AsyncRead + Unpin>(
        r: R,
    ) -> UnpackFuture<T, R, 256> {
        unpack_bufsz(r)
    }

    pub const fn unpack_bufsz<
        'a,
        const BUFSZ: usize,
        T: MufieldT + ~const MuStarve + 'a,
        R: AsyncRead + Unpin,
    >(
        r: R,
    ) -> UnpackFuture<T, R, BUFSZ> {
        UnpackFuture {
            r,
            req: T::min_starve(),
            _t: PhantomData,
            _bufsz: PhantomData,
        }
    }

    pub const fn pack<'a, T: MufieldT + 'a, W: AsyncWrite + Unpin>(
        t: &'a T,
        w: W,
    ) -> PackFuture<T, W> {
        PackFuture {
            w,
            objref: t,
            _t: PhantomData,
        }
    }

    #[pin_project]
    pub struct PackFuture<'a, T, W> {
        w: W,
        objref: &'a T,
        _t: PhantomData<T>,
    }

    impl<'a, T: MufieldT, W: AsyncWrite + Unpin> Future for PackFuture<'a, T, W> {
        type Output = tokio::io::Result<()>;

        fn poll(mut self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
            // TODO: Make enc return some kind of future that writes directly to the writer
            let mut buf = Vec::new();
            match self.objref.enc(&mut buf) {
                Ok(_) => (),
                Err(e) => return Poll::Ready(Err(e)),
            };
            match AsyncWrite::poll_write(Pin::new(&mut self.w), cx, &buf) {
                Poll::Ready(Ok(_)) => Poll::Ready(Ok(())),
                Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
                Poll::Pending => Poll::Pending,
            }
        }
    }
}

pub mod sync {
    use crate::{DecResult, MuError, MufieldT, WaitBuffer};
    use std::io;

    // TODO: pack functions that return iterator and stack vector
    pub fn pack<T: MufieldT, W: std::io::Write>(v: &T, w: &mut W) -> std::io::Result<()> {
        v.enc(w)
    }

    #[tracing::instrument(skip(r))]
    pub fn unpack<T: MufieldT, R: io::Read>(r: &mut R) -> DecResult<T> {
        let mut buf = WaitBuffer::new(r);
        loop {
            match T::dec(&mut buf) {
                Ok(r) => return Ok(r),
                Err(MuError::StarveMin(min)) => {
                    loop {
                        match buf.ready(min) {
                            Ok(_) => break,
                            Err(e) => match e.kind() {
                                io::ErrorKind::WouldBlock => (),
                                _ => return Err(MuError::Io(e)),
                            },
                        }
                    }
                    continue;
                }
                Err(e) => return Err(e),
            };
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PrimitiveIdx(pub usize);

#[derive(Debug, Clone)]
pub struct IntPrimitive {
    pub val: u64,
    pub max: u64,
}

#[derive(Debug, Clone)]
pub struct VectorPrimitive {
    len: PrimitiveIdx,
    ptr: Vec<Primitive>,
}

#[derive(Debug, Clone)]
pub enum Primitive {
    Int(IntPrimitive),
    Vector(VectorPrimitive),
}

impl MufieldT for Primitive {
    fn enc<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        match self {
            Primitive::Int(int) => {
                let imax = int.max;
                let ival = int.val;
                let range_u8 = ..(u8::MAX as u64);
                let range_u16 = ..(u16::MAX as u64);
                let range_u32 = ..(u32::MAX as u64);
                let range_u64 = ..u64::MAX;

                if range_u8.contains(&imax) {
                    w.write_all(&(ival as u8).to_be_bytes())
                } else if range_u16.contains(&imax) {
                    w.write_all(&(ival as u16).to_be_bytes())
                } else if range_u32.contains(&imax) {
                    w.write_all(&(ival as u32).to_be_bytes())
                } else if range_u64.contains(&imax) {
                    w.write_all(&ival.to_be_bytes())
                } else {
                    unreachable!()
                }
            }
            Primitive::Vector(vec) => {
                for p in &vec.ptr {
                    p.enc(w)?;
                }
                todo!()
            }
        }
    }

    fn dec<R: std::io::Read>(r: &mut R) -> DecResult<Self> {
        todo!()
    }
}

impl const MuStarve for Primitive {
    fn min_starve() -> usize {
        // FIXME: Return Infinity
        todo!()
    }
}

pub trait PrimitiveExpand {
    fn expand(&self, out: &mut Vec<Primitive>);
}

pub use conv::*;

pub mod conv {
    use std::ops::Sub;

    use num::Integer;

    use crate::{IntPrimitive, Primitive};

    pub fn i2max<Int: Integer + Copy + std::fmt::Debug>(a_max: Int, b_max: Int) -> Int
    where
        Int: Sub<Output = Int>,
    {
        let a_max1 = a_max.sub(Int::one());
        let b_max1 = b_max.sub(Int::one());
        i2merge::<Int>(a_max1, a_max, b_max1) + Int::one()
    }

    pub fn i2merge<Int: Integer + Copy>(a: Int, a_max: Int, b: Int) -> Int {
        // where a < m;
        // o=a+(mb)
        //
        // a=o%m
        // b=(o-a)/m

        debug_assert!(a < a_max);
        a + (a_max * b)
    }

    pub fn i2split<Int: Integer + Copy>(o: Int, a_max: Int) -> (Int, Int) {
        let a = o % a_max;
        let b = (o - a) / a_max;
        (a, b)
    }

    pub fn combine_ints(arr: &mut Vec<Primitive>) {
        loop {
            // FIXME: WHY DO I HAVE TO CLONE HERE????????????????????????????????????
            let ints = arr
                .clone()
                .into_iter()
                .enumerate()
                .filter_map(|(i, p)| match p {
                    Primitive::Int(intp) => Some((i, intp)),
                    _ => None,
                });

            let (smallint_i, smallint) =
                match ints.clone().min_by(|(_ix, x), (_iy, y)| x.max.cmp(&y.max)) {
                    Some(x) => x,
                    None => break,
                };
            let (bigint_i, bigint) = match ints
                .clone()
                .filter(|(ix, _x)| *ix != smallint_i)
                .max_by(|(_ix, x), (_iy, y)| x.max.cmp(&y.max))
            {
                Some(x) => x,
                None => break,
            };

            debug_assert_ne!(smallint_i, bigint_i);

            drop(ints);

            let combined_val = i2merge(smallint.val, smallint.max, bigint.val);
            let combined_max = i2max(smallint.max, bigint.max);
            debug_assert_eq!(
                i2split(combined_val, smallint.max),
                (smallint.val, bigint.val)
            );

            let combined_primitive = IntPrimitive {
                val: combined_val,
                max: combined_max,
            };

            arr[smallint_i] = Primitive::Int(combined_primitive);
            arr.swap_remove(bigint_i);

            debug_assert!(combined_val < combined_max);
        }
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::AsyncRead;
    use tracing::Level;

    use crate::{self as mupack, Hello, MufieldT, PrimitiveExpand};
    use std::fmt::Debug;
    use std::io;

    #[test]
    fn primitives() {
        let subscriber = tracing_subscriber::FmtSubscriber::builder()
            .with_test_writer()
            .with_max_level(Level::TRACE)
            .finish();
        let _tracing = tracing::subscriber::set_default(subscriber);

        fn test_arr(mut list: Vec<mupack::Primitive>) {
            mupack::conv::combine_ints(&mut list);
            tracing::trace!(primitives = ?&list);

            let mut buf = Vec::new();
            for p in list {
                p.enc(&mut buf).unwrap();
            }
            tracing::trace!(out = ?&buf);
        }

        test_arr(vec![
            mupack::Primitive::Int(mupack::IntPrimitive { val: 4, max: 5 }),
            mupack::Primitive::Int(mupack::IntPrimitive { val: 9, max: 10 }),
            mupack::Primitive::Int(mupack::IntPrimitive { val: 240, max: 256 }),
        ]);
        let mut hello_expand = Vec::new();
        Hello {
            byte: 2,
            maybe_exists: Some(140),
        }
        .expand(&mut hello_expand);
        test_arr(hello_expand);
    }

    #[tokio::test]
    async fn test() {
        let subscriber = tracing_subscriber::FmtSubscriber::builder()
            .with_test_writer()
            .with_max_level(Level::TRACE)
            .finish();
        let _tracing = tracing::subscriber::set_default(subscriber);

        async fn with_reader<R: AsyncRead + Unpin, T: PartialEq + MufieldT + Debug>(
            mut r: R,
            orig: T,
        ) {
            let out = mupack::nonblock::unpack(&mut r).await.unwrap();
            assert_eq!(orig, out);
        }

        let orig = Hello {
            byte: 2,
            maybe_exists: Some(69),
        };

        let mut out = Vec::new();
        mupack::sync::pack(&orig, &mut out).unwrap();

        with_reader(io::Cursor::new(out.clone()), orig).await;
    }

    #[test]
    pub fn i2merge() {
        let (a, a_max) = (4, 5);
        let b = 20;

        let o = mupack::i2merge(a, a_max, b);

        let (a_out, b_out) = mupack::i2split(o, a_max);
        assert_eq!(a, a_out);
        assert_eq!(b, b_out);
    }
}
