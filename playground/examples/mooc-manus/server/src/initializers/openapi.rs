use async_trait::async_trait;
use axum::{http::header, routing::get, Json, Router};
use loco_rs::{
    app::{AppContext, Initializer},
    config::{JWTLocation, JWTLocationConfig},
    Error, Result,
};
use serde::Deserialize;
use utoipa::openapi::security::{ApiKey, ApiKeyValue, HttpAuthScheme, HttpBuilder, SecurityScheme};
use utoipa_redoc::{Redoc, Servable};
use utoipa_swagger_ui::SwaggerUi;

pub struct OpenApiInitializer;

#[derive(Deserialize)]
struct OpenApiConfig {
    swagger: Option<UiConfig>,
    redoc: Option<UiConfig>,
}

#[derive(Deserialize)]
struct UiConfig {
    url: String,
    spec_json_url: Option<String>,
    spec_yaml_url: Option<String>,
}

#[async_trait]
impl Initializer for OpenApiInitializer {
    fn name(&self) -> String {
        "openapi".to_string()
    }

    async fn after_routes(&self, mut router: Router, ctx: &AppContext) -> Result<Router> {
        let Some(config) = ctx
            .config
            .initializers
            .as_ref()
            .and_then(|map| map.get("openapi"))
        else {
            return Ok(router);
        };
        let config: OpenApiConfig = serde_json::from_value(config.clone())?;
        let mut document = crate::openapi::document();

        // 安全方案直接读取当前应用的 JWT 配置。
        let location = ctx
            .config
            .auth
            .as_ref()
            .and_then(|auth| auth.jwt.as_ref())
            .and_then(|jwt| jwt.location.as_ref())
            .and_then(|location| match location {
                JWTLocationConfig::Single(location) => Some(location),
                JWTLocationConfig::Multiple(locations) => locations.first(),
            })
            .unwrap_or(&JWTLocation::Bearer);
        let jwt = match location {
            JWTLocation::Bearer => SecurityScheme::Http(
                HttpBuilder::new()
                    .scheme(HttpAuthScheme::Bearer)
                    .bearer_format("JWT")
                    .build(),
            ),
            JWTLocation::Query { name } => {
                SecurityScheme::ApiKey(ApiKey::Query(ApiKeyValue::new(name)))
            }
            JWTLocation::Cookie { name } => {
                SecurityScheme::ApiKey(ApiKey::Cookie(ApiKeyValue::new(name)))
            }
        };
        document
            .components
            .get_or_insert_with(Default::default)
            .add_security_schemes_from_iter([
                ("jwt_token", jwt),
                (
                    "api_key",
                    SecurityScheme::ApiKey(ApiKey::Header(ApiKeyValue::new("apikey"))),
                ),
            ]);

        if let Some(swagger) = &config.swagger {
            let json_url = swagger.spec_json_url.as_ref().ok_or_else(|| {
                Error::Message("initializers.openapi.swagger.spec_json_url is required".into())
            })?;
            router = router
                .merge(SwaggerUi::new(swagger.url.clone()).url(json_url.clone(), document.clone()));
        }
        if let Some(redoc) = &config.redoc {
            router = router.merge(Redoc::with_url(redoc.url.clone(), document.clone()));
            if let Some(json_url) = &redoc.spec_json_url {
                let spec = document.clone();
                router = router.route(
                    json_url,
                    get(move || {
                        let spec = spec.clone();
                        async move { Json(spec) }
                    }),
                );
            }
        }
        for ui in [config.swagger, config.redoc].into_iter().flatten() {
            if let Some(yaml_url) = ui.spec_yaml_url {
                let yaml = document
                    .to_yaml()
                    .map_err(|err| Error::Any(Box::new(err)))?;
                router = router.route(
                    &yaml_url,
                    get(move || {
                        let yaml = yaml.clone();
                        async move { ([(header::CONTENT_TYPE, "application/yaml")], yaml) }
                    }),
                );
            }
        }
        Ok(router)
    }
}
