pub mod app;

pub mod application;
pub mod controllers;
pub mod data;
pub mod domain;
pub mod infrastructure;
pub mod initializers;
pub mod interfaces;
pub mod mailers;
pub mod models;
pub mod openapi;
pub mod tasks;
#[cfg(test)]
#[path = "../tests/support/file_database.rs"]
pub(crate) mod test_database;
pub mod views;
pub mod workers;
