use std::{collections::HashMap, io};

type Result<T> = io::Result<T>;

trait MufieldT: Sized {
    fn enc<W: io::Write>(&self, w: &mut W) -> Result<()>;
    fn dec<R: io::Read>(r: &mut R) -> Result<Self>;
}

#[derive(Clone, Copy, Debug)]
struct MufieldIdx(usize);
#[derive(Debug)]
struct MubyteIdx(usize);

enum SizeData {
    Exact(usize),
    Unsized(UnknownSizeData),
}

#[derive(Debug)]
enum AllSizes {
    Finite(Vec<usize>),
    Infinite,
}

struct UnknownSizeData {
    /// My sizer fields
    mysz: Vec<MufieldIdx>,
    // sizes: AllSizes<Box<dyn Iterator<Item = usize>>>,
    sizes: AllSizes,
    /// Function which determines the size of this field depending on the sizers
    size_fn: fn(&Mudata, &[MufieldIdx]) -> usize,
}

struct Mufield {
    /// Name of field
    name: &'static str,
    /// Is this field a sizer for another field?
    isz: Option<MufieldIdx>,
    /// Size information about this field
    sz: SizeData,
    /// Indexes into the mudata array for each byte, in order.
    byteidx: Vec<MubyteIdx>,
}

struct Mufmt {
    fields: Vec<Mufield>,
}

#[derive(Clone)]
struct Mudata {
    bytes: Vec<(MufieldIdx, u8)>,
}

trait Mu {
    fn mufmt() -> Mufmt;
    fn mudata(&self) -> Mudata;
    fn muaccept(mudata: MudataInput) -> Result<Self>
    where
        Self: Sized;
}

struct Hello {
    byte: u8,
    flag: bool,
    /// Exists only if `flag` is set to true.
    maybe_exists: Option<u8>,
}

struct MudataInput {
    map: HashMap<&'static str, Vec<u8>>,
}

fn hello_maybe_exists_szfn(data: &Mudata, fieldidx: &[MufieldIdx]) -> usize {
    let (_, byte) = data.bytes[fieldidx[0].0];
    /*if byte != 0 {
        1
    } else {
        0
    }*/
    byte as usize
}

impl Mu for Hello {
    fn mufmt() -> Mufmt {
        Mufmt {
            fields: vec![
                Mufield {
                    name: "byte",
                    isz: None,
                    sz: SizeData::Exact(1),
                    byteidx: vec![MubyteIdx(0)],
                },
                Mufield {
                    name: "flag",
                    isz: Some(MufieldIdx(2)),
                    sz: SizeData::Exact(1),
                    byteidx: vec![MubyteIdx(1)],
                },
                Mufield {
                    name: "maybe_exists",
                    isz: None,
                    sz: SizeData::Unsized(UnknownSizeData {
                        mysz: vec![MufieldIdx(1)],
                        size_fn: hello_maybe_exists_szfn,
                        sizes: AllSizes::Finite(vec![0, 1]),
                    }),
                    byteidx: vec![MubyteIdx(2)],
                },
            ],
        }
    }

    fn mudata(&self) -> Mudata {
        Mudata {
            bytes: vec![
                (MufieldIdx(0), self.byte),
                (MufieldIdx(1), self.flag as u8),
                (MufieldIdx(2), self.maybe_exists.unwrap_or(0)),
            ],
        }
    }

    fn muaccept(input: MudataInput) -> Result<Self>
    where
        Self: Sized,
    {
        let byte = input.map["byte"][0];
        let flag = input.map["flag"][0] != 0;
        let maybe_exists = if flag {
            Some(input.map["maybe_exists"][0])
        } else {
            assert_eq!(input.map["maybe_exists"].len(), 0);
            None
        };
        Ok(Hello {
            byte,
            flag,
            maybe_exists,
        })
    }
}

enum MaxLen {
    Infinite,
    Finite(usize),
}

fn max_len(fmt: Mufmt) -> MaxLen {
    MaxLen::Finite(fmt.fields.iter().map(|f| f.byteidx.len()).sum())
}

/// Encode the struct into a stream of bytes
fn pack<M: Mu>(struct_: M) -> Vec<u8> {
    // Not worth trying to determine capacity before speed-wise
    // (I think, didn't verify)

    let fmt = M::mufmt();
    let mut data = struct_.mudata();
    let data_clone = data.clone();

    let mut curbyteidx = 0;
    data.bytes.retain(|(MufieldIdx(idx), _)| {
        let field = &fmt.fields[*idx];
        let sz = match &field.sz {
            SizeData::Exact(n) => *n,
            SizeData::Unsized(un) => {
                let sz = (un.size_fn)(&data_clone, &un.mysz);
                #[cfg(debug_assertions)]
                {
                    assert!(match &un.sizes {
                        AllSizes::Finite(szs) => {
                            szs.contains(&sz)
                        }
                        AllSizes::Infinite => true,
                    })
                }
                sz
            }
        };
        let (n_byte, _) = field.byteidx.iter().enumerate().find(|(_n_byte, MubyteIdx(bx))| *bx == curbyteidx).unwrap();
        curbyteidx += 1;

        dbg!(sz);
        dbg!(n_byte);
        n_byte < sz
    });

    data.bytes.sort_by(|(a_idx, _), (b_idx, _)| {
        fmt.fields[a_idx.0]
            .isz
            .is_some()
            .cmp(&fmt.fields[b_idx.0].isz.is_some())
            .reverse()
    });

    let mut out = Vec::new();
    for (_, byte) in data.bytes {
        out.push(byte);
    }

    dbg!(&out);
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        crate::pack(crate::Hello {
            byte: 69,
            flag: false,
            maybe_exists: Some(2),
        });
        eprintln!("Done.");
    }
}
