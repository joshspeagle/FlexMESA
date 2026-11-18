// SPDX-License-Identifier: LGPL-3.0-or-later
//! Deterministic IEEE-754 little-endian binary `serde` codec for exact
//! checkpoint serialization without hand-enumerated field codecs.

use std::fmt;

use serde::de::{
    self, DeserializeOwned, DeserializeSeed, EnumAccess, MapAccess, SeqAccess,
    VariantAccess, Visitor,
};
use serde::ser::{
    self, Serialize, SerializeMap, SerializeSeq, SerializeStruct,
    SerializeStructVariant, SerializeTuple, SerializeTupleStruct,
    SerializeTupleVariant,
};

/// Errors produced by [`to_bytes`] and [`from_bytes`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinCodecError {
    /// Serializer or deserializer hit a custom `serde` error message.
    Message(String),
    /// Input byte slice ended before a complete value could be decoded.
    UnexpectedEof,
    /// Trailing unconsumed bytes remained after decoding the root value.
    TrailingBytes {
        /// Number of trailing bytes after the decoded payload.
        remaining: usize,
    },
    /// Malformed discriminant, sequence length, boolean, char, or UTF-8 string.
    InvalidEncoding(String),
    /// Sequence/map length missing during serialization.
    UnknownLength,
}

impl fmt::Display for BinCodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Message(msg) => f.write_str(msg),
            Self::UnexpectedEof => {
                f.write_str("unexpected end of binary stream")
            }
            Self::TrailingBytes { remaining } => write!(
                f,
                "trailing garbage bytes in binary stream ({remaining} bytes)"
            ),
            Self::InvalidEncoding(msg) => {
                write!(f, "invalid binary encoding: {msg}")
            }
            Self::UnknownLength => {
                f.write_str("binary codec requires known-length sequences/maps")
            }
        }
    }
}

impl std::error::Error for BinCodecError {}

impl ser::Error for BinCodecError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Self::Message(msg.to_string())
    }
}

impl de::Error for BinCodecError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Self::Message(msg.to_string())
    }
}

/// Encodes `value` into deterministic IEEE-754 little-endian binary bytes.
///
/// # Errors
/// Returns [`BinCodecError`] if `value` contains unknown-length sequences/maps
/// or fails `Serialize`.
pub fn to_bytes<T: Serialize + ?Sized>(
    value: &T,
) -> Result<Vec<u8>, BinCodecError> {
    let mut ser = BinSerializer {
        out: Vec::with_capacity(256),
    };
    value.serialize(&mut ser)?;
    Ok(ser.out)
}

/// Decodes an instance of `T` from exact bytes produced by [`to_bytes`],
/// rejecting truncated inputs or trailing unconsumed bytes.
///
/// # Errors
/// Returns [`BinCodecError`] on EOF, malformed encoding, or trailing bytes.
pub fn from_bytes<T: DeserializeOwned>(
    bytes: &[u8],
) -> Result<T, BinCodecError> {
    let mut de = BinDeserializer {
        input: bytes,
        pos: 0,
    };
    let val = T::deserialize(&mut de)?;
    if de.pos != de.input.len() {
        return Err(BinCodecError::TrailingBytes {
            remaining: de.input.len() - de.pos,
        });
    }
    Ok(val)
}

struct BinSerializer {
    out: Vec<u8>,
}

impl BinSerializer {
    #[inline]
    fn write_len(&mut self, len: usize) {
        let len_u64 = u64::try_from(len).unwrap_or(u64::MAX);
        self.out.extend_from_slice(&len_u64.to_le_bytes());
    }
}

impl ser::Serializer for &mut BinSerializer {
    type Ok = ();
    type Error = BinCodecError;
    type SerializeSeq = Self;
    type SerializeTuple = Self;
    type SerializeTupleStruct = Self;
    type SerializeTupleVariant = Self;
    type SerializeMap = Self;
    type SerializeStruct = Self;
    type SerializeStructVariant = Self;

    #[inline]
    fn serialize_bool(self, v: bool) -> Result<(), Self::Error> {
        self.out.push(u8::from(v));
        Ok(())
    }

    #[inline]
    fn serialize_i8(self, v: i8) -> Result<(), Self::Error> {
        self.out.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn serialize_i16(self, v: i16) -> Result<(), Self::Error> {
        self.out.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn serialize_i32(self, v: i32) -> Result<(), Self::Error> {
        self.out.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn serialize_i64(self, v: i64) -> Result<(), Self::Error> {
        self.out.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn serialize_i128(self, v: i128) -> Result<(), Self::Error> {
        self.out.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn serialize_u8(self, v: u8) -> Result<(), Self::Error> {
        self.out.push(v);
        Ok(())
    }

    #[inline]
    fn serialize_u16(self, v: u16) -> Result<(), Self::Error> {
        self.out.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn serialize_u32(self, v: u32) -> Result<(), Self::Error> {
        self.out.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn serialize_u64(self, v: u64) -> Result<(), Self::Error> {
        self.out.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn serialize_u128(self, v: u128) -> Result<(), Self::Error> {
        self.out.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn serialize_f32(self, v: f32) -> Result<(), Self::Error> {
        self.out.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn serialize_f64(self, v: f64) -> Result<(), Self::Error> {
        self.out.extend_from_slice(&v.to_le_bytes());
        Ok(())
    }

    #[inline]
    fn serialize_char(self, v: char) -> Result<(), Self::Error> {
        self.out.extend_from_slice(&u32::from(v).to_le_bytes());
        Ok(())
    }

    #[inline]
    fn serialize_str(self, v: &str) -> Result<(), Self::Error> {
        self.write_len(v.len());
        self.out.extend_from_slice(v.as_bytes());
        Ok(())
    }

    #[inline]
    fn serialize_bytes(self, v: &[u8]) -> Result<(), Self::Error> {
        self.write_len(v.len());
        self.out.extend_from_slice(v);
        Ok(())
    }

    #[inline]
    fn serialize_none(self) -> Result<(), Self::Error> {
        self.out.push(0);
        Ok(())
    }

    #[inline]
    fn serialize_some<T: Serialize + ?Sized>(
        self,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.out.push(1);
        value.serialize(self)
    }

    #[inline]
    fn serialize_unit(self) -> Result<(), Self::Error> {
        Ok(())
    }

    #[inline]
    fn serialize_unit_struct(
        self,
        _name: &'static str,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    #[inline]
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
    ) -> Result<(), Self::Error> {
        self.serialize_u32(variant_index)
    }

    #[inline]
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        value.serialize(self)
    }

    #[inline]
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.serialize_u32(variant_index)?;
        value.serialize(self)
    }

    fn serialize_seq(
        self,
        len: Option<usize>,
    ) -> Result<Self::SerializeSeq, Self::Error> {
        let len = len.ok_or(BinCodecError::UnknownLength)?;
        self.write_len(len);
        Ok(self)
    }

    fn serialize_tuple(
        self,
        _len: usize,
    ) -> Result<Self::SerializeTuple, Self::Error> {
        Ok(self)
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        Ok(self)
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        self.serialize_u32(variant_index)?;
        Ok(self)
    }

    fn serialize_map(
        self,
        len: Option<usize>,
    ) -> Result<Self::SerializeMap, Self::Error> {
        let len = len.ok_or(BinCodecError::UnknownLength)?;
        self.write_len(len);
        Ok(self)
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        Ok(self)
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        self.serialize_u32(variant_index)?;
        Ok(self)
    }
}

impl SerializeSeq for &mut BinSerializer {
    type Ok = ();
    type Error = BinCodecError;
    #[inline]
    fn serialize_element<T: Serialize + ?Sized>(
        &mut self,
        value: &T,
    ) -> Result<(), Self::Error> {
        value.serialize(&mut **self)
    }
    #[inline]
    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl SerializeTuple for &mut BinSerializer {
    type Ok = ();
    type Error = BinCodecError;
    #[inline]
    fn serialize_element<T: Serialize + ?Sized>(
        &mut self,
        value: &T,
    ) -> Result<(), Self::Error> {
        value.serialize(&mut **self)
    }
    #[inline]
    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl SerializeTupleStruct for &mut BinSerializer {
    type Ok = ();
    type Error = BinCodecError;
    #[inline]
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        value: &T,
    ) -> Result<(), Self::Error> {
        value.serialize(&mut **self)
    }
    #[inline]
    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl SerializeTupleVariant for &mut BinSerializer {
    type Ok = ();
    type Error = BinCodecError;
    #[inline]
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        value: &T,
    ) -> Result<(), Self::Error> {
        value.serialize(&mut **self)
    }
    #[inline]
    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl SerializeMap for &mut BinSerializer {
    type Ok = ();
    type Error = BinCodecError;
    #[inline]
    fn serialize_key<T: Serialize + ?Sized>(
        &mut self,
        key: &T,
    ) -> Result<(), Self::Error> {
        key.serialize(&mut **self)
    }
    #[inline]
    fn serialize_value<T: Serialize + ?Sized>(
        &mut self,
        value: &T,
    ) -> Result<(), Self::Error> {
        value.serialize(&mut **self)
    }
    #[inline]
    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl SerializeStruct for &mut BinSerializer {
    type Ok = ();
    type Error = BinCodecError;
    #[inline]
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        value.serialize(&mut **self)
    }
    #[inline]
    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}

impl SerializeStructVariant for &mut BinSerializer {
    type Ok = ();
    type Error = BinCodecError;
    #[inline]
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        value.serialize(&mut **self)
    }
    #[inline]
    fn end(self) -> Result<(), Self::Error> {
        Ok(())
    }
}

struct BinDeserializer<'de> {
    input: &'de [u8],
    pos: usize,
}

impl<'de> BinDeserializer<'de> {
    #[inline]
    fn read_bytes(&mut self, n: usize) -> Result<&'de [u8], BinCodecError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(BinCodecError::UnexpectedEof)?;
        let slice = self
            .input
            .get(self.pos..end)
            .ok_or(BinCodecError::UnexpectedEof)?;
        self.pos = end;
        Ok(slice)
    }

    #[inline]
    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], BinCodecError> {
        let slice = self.read_bytes(N)?;
        let mut arr = [0_u8; N];
        arr.copy_from_slice(slice);
        Ok(arr)
    }

    #[inline]
    fn read_u64_len(&mut self) -> Result<usize, BinCodecError> {
        let raw = u64::from_le_bytes(self.read_array::<8>()?);
        let len = usize::try_from(raw).map_err(|_| {
            BinCodecError::InvalidEncoding(format!(
                "sequence length {raw} exceeds platform usize"
            ))
        })?;
        let remaining = self.input.len() - self.pos;
        if len > remaining && len > 1_048_576 {
            return Err(BinCodecError::InvalidEncoding(format!(
                "corrupt sequence length {len} exceeds remaining {remaining} bytes"
            )));
        }
        Ok(len)
    }
}

impl<'de> de::Deserializer<'de> for &mut BinDeserializer<'de> {
    type Error = BinCodecError;

    fn deserialize_any<V: Visitor<'de>>(
        self,
        _visitor: V,
    ) -> Result<V::Value, Self::Error> {
        Err(BinCodecError::InvalidEncoding(String::from(
            "binary codec is not self-describing",
        )))
    }

    #[inline]
    fn deserialize_bool<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        match self.read_bytes(1)?[0] {
            0 => visitor.visit_bool(false),
            1 => visitor.visit_bool(true),
            b => Err(BinCodecError::InvalidEncoding(format!(
                "invalid bool tag {b}"
            ))),
        }
    }

    #[inline]
    fn deserialize_i8<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_i8(i8::from_le_bytes(self.read_array::<1>()?))
    }

    #[inline]
    fn deserialize_i16<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_i16(i16::from_le_bytes(self.read_array::<2>()?))
    }

    #[inline]
    fn deserialize_i32<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_i32(i32::from_le_bytes(self.read_array::<4>()?))
    }

    #[inline]
    fn deserialize_i64<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_i64(i64::from_le_bytes(self.read_array::<8>()?))
    }

    #[inline]
    fn deserialize_i128<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_i128(i128::from_le_bytes(self.read_array::<16>()?))
    }

    #[inline]
    fn deserialize_u8<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_u8(self.read_bytes(1)?[0])
    }

    #[inline]
    fn deserialize_u16<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_u16(u16::from_le_bytes(self.read_array::<2>()?))
    }

    #[inline]
    fn deserialize_u32<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_u32(u32::from_le_bytes(self.read_array::<4>()?))
    }

    #[inline]
    fn deserialize_u64<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_u64(u64::from_le_bytes(self.read_array::<8>()?))
    }

    #[inline]
    fn deserialize_u128<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_u128(u128::from_le_bytes(self.read_array::<16>()?))
    }

    #[inline]
    fn deserialize_f32<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_f32(f32::from_le_bytes(self.read_array::<4>()?))
    }

    #[inline]
    fn deserialize_f64<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_f64(f64::from_le_bytes(self.read_array::<8>()?))
    }

    #[inline]
    fn deserialize_char<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        let code = u32::from_le_bytes(self.read_array::<4>()?);
        let ch = char::from_u32(code).ok_or_else(|| {
            BinCodecError::InvalidEncoding(format!(
                "invalid scalar char {code}"
            ))
        })?;
        visitor.visit_char(ch)
    }

    #[inline]
    fn deserialize_str<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        let len = self.read_u64_len()?;
        let raw = self.read_bytes(len)?;
        let s = std::str::from_utf8(raw).map_err(|e| {
            BinCodecError::InvalidEncoding(format!("invalid UTF-8 string: {e}"))
        })?;
        visitor.visit_borrowed_str(s)
    }

    #[inline]
    fn deserialize_string<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_str(visitor)
    }

    #[inline]
    fn deserialize_bytes<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        let len = self.read_u64_len()?;
        let raw = self.read_bytes(len)?;
        visitor.visit_borrowed_bytes(raw)
    }

    #[inline]
    fn deserialize_byte_buf<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_bytes(visitor)
    }

    #[inline]
    fn deserialize_option<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        match self.read_bytes(1)?[0] {
            0 => visitor.visit_none(),
            1 => visitor.visit_some(self),
            b => Err(BinCodecError::InvalidEncoding(format!(
                "invalid option tag {b}"
            ))),
        }
    }

    #[inline]
    fn deserialize_unit<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_unit()
    }

    #[inline]
    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_unit()
    }

    #[inline]
    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_newtype_struct(self)
    }

    #[inline]
    fn deserialize_seq<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        let len = self.read_u64_len()?;
        visitor.visit_seq(FixedSeqAccess {
            de: self,
            remaining: len,
        })
    }

    #[inline]
    fn deserialize_tuple<V: Visitor<'de>>(
        self,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_seq(FixedSeqAccess {
            de: self,
            remaining: len,
        })
    }

    #[inline]
    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_tuple(len, visitor)
    }

    #[inline]
    fn deserialize_map<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        let len = self.read_u64_len()?;
        visitor.visit_map(FixedMapAccess {
            de: self,
            remaining: len,
        })
    }

    #[inline]
    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.deserialize_tuple(fields.len(), visitor)
    }

    #[inline]
    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        visitor.visit_enum(BinEnumAccess { de: self })
    }

    #[inline]
    fn deserialize_identifier<V: Visitor<'de>>(
        self,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        let idx = u32::from_le_bytes(self.read_array::<4>()?);
        visitor.visit_u32(idx)
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(
        self,
        _visitor: V,
    ) -> Result<V::Value, Self::Error> {
        Err(BinCodecError::InvalidEncoding(String::from(
            "ignored fields are unsupported in binary codec",
        )))
    }
}

struct FixedSeqAccess<'a, 'de> {
    de: &'a mut BinDeserializer<'de>,
    remaining: usize,
}

impl<'de> SeqAccess<'de> for FixedSeqAccess<'_, 'de> {
    type Error = BinCodecError;

    #[inline]
    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, Self::Error> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        seed.deserialize(&mut *self.de).map(Some)
    }

    #[inline]
    fn size_hint(&self) -> Option<usize> {
        Some(self.remaining.min(4096))
    }
}

struct FixedMapAccess<'a, 'de> {
    de: &'a mut BinDeserializer<'de>,
    remaining: usize,
}

impl<'de> MapAccess<'de> for FixedMapAccess<'_, 'de> {
    type Error = BinCodecError;

    #[inline]
    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, Self::Error> {
        if self.remaining == 0 {
            return Ok(None);
        }
        self.remaining -= 1;
        seed.deserialize(&mut *self.de).map(Some)
    }

    #[inline]
    fn next_value_seed<V: DeserializeSeed<'de>>(
        &mut self,
        seed: V,
    ) -> Result<V::Value, Self::Error> {
        seed.deserialize(&mut *self.de)
    }
}

struct BinEnumAccess<'a, 'de> {
    de: &'a mut BinDeserializer<'de>,
}

impl<'de> EnumAccess<'de> for BinEnumAccess<'_, 'de> {
    type Error = BinCodecError;
    type Variant = Self;

    #[inline]
    fn variant_seed<V: DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant), Self::Error> {
        let val = seed.deserialize(&mut *self.de)?;
        Ok((val, self))
    }
}

impl<'de> VariantAccess<'de> for BinEnumAccess<'_, 'de> {
    type Error = BinCodecError;

    #[inline]
    fn unit_variant(self) -> Result<(), Self::Error> {
        Ok(())
    }

    #[inline]
    fn newtype_variant_seed<T: DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> Result<T::Value, Self::Error> {
        seed.deserialize(self.de)
    }

    #[inline]
    fn tuple_variant<V: Visitor<'de>>(
        self,
        len: usize,
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        de::Deserializer::deserialize_tuple(self.de, len, visitor)
    }

    #[inline]
    fn struct_variant<V: Visitor<'de>>(
        self,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        de::Deserializer::deserialize_struct(self.de, "", fields, visitor)
    }
}
