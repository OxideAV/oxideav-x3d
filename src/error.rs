//! Crate-local error type.
//!
//! The document layer (XML / ClassicVRML parsing, field-value parsing,
//! prototype expansion) reports [`Error`]; the
//! [`Mesh3DDecoder`](oxideav_mesh3d::Mesh3DDecoder) /
//! [`Mesh3DEncoder`](oxideav_mesh3d::Mesh3DEncoder) impls map it onto
//! the `oxideav-mesh3d` error vocabulary at the trait boundary (which
//! itself is `oxideav_core::Error` under the `registry` feature).

use std::fmt;

/// Errors produced while reading or writing X3D.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Lexical or syntactic violation of the encoding (XML
    /// well-formedness, ClassicVRML grammar). `line` / `column` are
    /// 1-based and point at the offending input.
    Syntax {
        /// 1-based line of the offending input.
        line: usize,
        /// 1-based column of the offending input.
        column: usize,
        /// Human-readable description.
        message: String,
    },
    /// A field value could not be parsed as its declared type.
    Field {
        /// Node type owning the field (or `"field"` for declarations).
        node: String,
        /// Field name.
        field: String,
        /// Human-readable description.
        message: String,
    },
    /// Structurally invalid document (missing `<Scene>`, not an X3D
    /// file, bad gzip stream, ...).
    Invalid(String),
    /// A configured [`Limits`](crate::Limits) cap was exceeded — the
    /// hostile-input guard fired.
    LimitExceeded(String),
    /// Construct this crate does not handle.
    Unsupported(String),
}

impl Error {
    /// Shorthand for [`Error::Invalid`].
    pub fn invalid(msg: impl Into<String>) -> Self {
        Self::Invalid(msg.into())
    }

    /// Shorthand for [`Error::LimitExceeded`].
    pub fn limit(msg: impl Into<String>) -> Self {
        Self::LimitExceeded(msg.into())
    }

    /// Shorthand for [`Error::Unsupported`].
    pub fn unsupported(msg: impl Into<String>) -> Self {
        Self::Unsupported(msg.into())
    }

    /// Map onto the `oxideav-mesh3d` error type (the trait boundary).
    pub fn into_mesh3d(self) -> oxideav_mesh3d::Error {
        match self {
            Self::Unsupported(m) => oxideav_mesh3d::Error::unsupported(format!("X3D: {m}")),
            other => oxideav_mesh3d::Error::invalid(format!("X3D: {other}")),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntax {
                line,
                column,
                message,
            } => write!(f, "syntax error at {line}:{column}: {message}"),
            Self::Field {
                node,
                field,
                message,
            } => write!(f, "bad value for {node}.{field}: {message}"),
            Self::Invalid(m) => write!(f, "invalid document: {m}"),
            Self::LimitExceeded(m) => write!(f, "limit exceeded: {m}"),
            Self::Unsupported(m) => write!(f, "unsupported: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<Error> for oxideav_mesh3d::Error {
    fn from(e: Error) -> Self {
        e.into_mesh3d()
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
