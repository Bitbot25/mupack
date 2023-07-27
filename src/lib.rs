use std::io;

#[derive(Debug)]
pub enum MuError {
    Io(std::io::Error),
    StarveMin(usize),
}

pub type DecResult<T> = std::result::Result<T, MuError>;

pub trait MufieldT: Sized {
    fn enc<W: io::Write>(&self, w: &mut W) -> io::Result<()>;
    fn dec<R: io::Read>(r: &mut R) -> DecResult<Self>;
}

struct Hello {
    byte: u8,
    maybe_exists: Option<u8>,
}

impl MufieldT for u8 {
    fn dec<R: io::Read>(r: &mut R) -> DecResult<Self> {
        let mut b = [0; 1];
        r.read_exact(&mut b).map_err(MuError::Io)?;
        Ok(b[0])
    }

    fn enc<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(&[*self])
    }
}

impl MufieldT for bool {
    fn enc<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        let b = *self as u8;
        u8::enc(&b, w)
    }

    fn dec<R: io::Read>(r: &mut R) -> DecResult<Self> {
        let b = u8::dec(r)?;
        Ok(b != 0)
    }
}

impl<T: MufieldT> MufieldT for Option<T> {
    fn enc<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        match self {
            Some(x) => {
                bool::enc(&true, w)?;
                T::enc(x, w)
            },
            None => {
                bool::enc(&false, w)
            }
        }
    }

    fn dec<R: io::Read>(r: &mut R) -> DecResult<Self> {
        let flag = bool::dec(r)?;
        if flag {
            let x = T::dec(r)?;
            Ok(Some(x))
        } else {
            Ok(None)
        }
    }
}

impl MufieldT for Hello {
    fn enc<W: io::Write>(&self, w: &mut W) -> io::Result<()> {
        u8::enc(&self.byte, w)?;
        Option::<u8>::enc(&self.maybe_exists, w)
    }

    fn dec<R: io::Read>(r: &mut R) -> DecResult<Self> {
        let byte = u8::dec(r)?;
        let maybe_exists = Option::<u8>::dec(r)?;
        Ok(Self { byte, maybe_exists })
    }
}

// TODO: pack functions that return iterator and stack vector
pub fn pack<T: MufieldT>(v: T) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    v.enc(&mut out)?; 
    Ok(out)
}

pub mod conv {
    pub struct ConstrainedInt<const MAX: u32>(pub u32);

    pub fn i2merge<const A_MAX: u32>(a: ConstrainedInt<A_MAX>, b: u32) -> u32 {
        // A_MAX=n, B_MAX=m
        // o=a+(mb)
        //
        // a=o%m
        // b=(o-a)/m

        a.0+(A_MAX*b)
    }

    pub fn i2split<const B_MAX: u32>(o: u32) -> (u32, u32) {
        dbg!(o);
        let a = o % B_MAX;
        dbg!(a);
        let b = (o - a) / B_MAX;
        dbg!(b);
        (a, b)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        dbg!(crate::pack(crate::Hello {
            byte: 69,
            maybe_exists: Some(2),
        }).unwrap());
        eprintln!("Done.");
    }

    #[test]
    pub fn i2merge() {
        let a = 4;
        let b = 20;
        let o = crate::conv::i2merge(crate::conv::ConstrainedInt::<5>(a), b);
        
        let (a_out, b_out) = crate::conv::i2split::<5>(o);
        assert_eq!(a, a_out);
        assert_eq!(b, b_out);
    }
}
