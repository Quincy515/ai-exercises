use std::{fs, path::PathBuf, process::Command};

use anyhow::Result;
use clap::{Parser, ValueEnum};
use crux_core::type_generation::facet::{Config, TypeRegistry};
use log::info;

use shared::AppCore;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum Language {
    Swift,
    Kotlin,
    Typescript,
}

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, value_enum)]
    language: Language,
    #[arg(short, long)]
    output_dir: PathBuf,
}

fn main() -> Result<()> {
    pretty_env_logger::init();
    let args = Args::parse();

    let typegen_app = TypeRegistry::new().register_app::<AppCore>()?.build()?;

    let name = match args.language {
        Language::Swift => "App",
        Language::Kotlin => "ai.lenexus.crux_template",
        Language::Typescript => "app",
    };
    let config = Config::builder(name, &args.output_dir).build();

    match args.language {
        Language::Swift => {
            info!("Typegen for Swift");
            typegen_app.swift(&config)?;
        }
        Language::Kotlin => {
            info!("Typegen for Kotlin");
            typegen_app.kotlin(&config)?;
        }
        Language::Typescript => {
            info!("Typegen for TypeScript");
            typegen_app.typescript(&config)?;

            // Keep generated source separate from the declarations consumed by strict shells.
            let tsconfig_path = args.output_dir.join("tsconfig.json");
            let mut tsconfig: serde_json::Value =
                serde_json::from_slice(&fs::read(&tsconfig_path)?)?;
            tsconfig["compilerOptions"]["outDir"] = "dist".into();
            tsconfig["exclude"] = serde_json::json!(["dist", "node_modules"]);
            fs::write(tsconfig_path, serde_json::to_vec_pretty(&tsconfig)?)?;

            let package_path = args.output_dir.join("package.json");
            let mut package: serde_json::Value = serde_json::from_slice(&fs::read(&package_path)?)?;
            package["files"] = serde_json::json!(["dist"]);
            package["exports"] = serde_json::json!({
                ".": { "types": "./dist/app.d.ts", "default": "./dist/app.js" },
                "./app": { "types": "./dist/app.d.ts", "default": "./dist/app.js" },
                "./bincode": { "types": "./dist/bincode/index.d.ts", "default": "./dist/bincode/index.js" },
                "./serde": { "types": "./dist/serde/index.d.ts", "default": "./dist/serde/index.js" },
                "./*.js": { "types": "./dist/*.d.ts", "default": "./dist/*.js" }
            });
            fs::write(package_path, serde_json::to_vec_pretty(&package)?)?;

            // crux_core 0.20 ignores its internal tsc exit status; propagate failures here.
            let status = Command::new("pnpm")
                .current_dir(&args.output_dir)
                .args(["exec", "tsc", "--build", "--force"])
                .status()?;
            anyhow::ensure!(status.success(), "TypeScript compilation failed: {status}");
        }
    }

    Ok(())
}
