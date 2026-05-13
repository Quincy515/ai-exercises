use crate::{
    mailers::auth::AuthMailer,
    models::{
        _entities::users,
        users::{LoginParams, RegisterParams},
    },
    views::auth::{CurrentResponse, LoginResponse},
};
use loco_openapi::prelude::{openapi, routes};
use loco_rs::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use utoipa::ToSchema;

pub static EMAIL_DOMAIN_RE: OnceLock<Regex> = OnceLock::new();

fn get_allow_email_domain_re() -> &'static Regex {
    EMAIL_DOMAIN_RE.get_or_init(|| {
        Regex::new(r"@example\.com$|@gmail\.com$").expect("Failed to compile regex")
    })
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct ForgotParams {
    pub email: String,
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct ResetParams {
    pub token: String,
    pub password: String,
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct MagicLinkParams {
    pub email: String,
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct ResendVerificationParams {
    pub email: String,
}

/// Register function creates a new user with the given parameters and sends a
/// welcome email to the user
#[utoipa::path(
    post,
    path = "/api/auth/register",
    tag = "认证",
    summary = "用户注册",
    description = "创建新用户并发送邮箱验证邮件。",
    request_body = RegisterParams,
    responses(
        (status = 200, description = "注册请求已接收")
    )
)]
#[debug_handler]
async fn register(
    State(ctx): State<AppContext>,
    Json(params): Json<RegisterParams>,
) -> Result<Response> {
    let res = users::Model::create_with_password(&ctx.db, &params).await;

    let user = match res {
        Ok(user) => user,
        Err(err) => {
            tracing::info!(
                message = err.to_string(),
                user_email = &params.email,
                "could not register user",
            );
            return format::json(());
        }
    };

    let user = user
        .into_active_model()
        .set_email_verification_sent(&ctx.db)
        .await?;

    AuthMailer::send_welcome(&ctx, &user).await?;

    format::json(())
}

/// Verify register user. if the user not verified his email, he can't login to
/// the system.
#[utoipa::path(
    get,
    path = "/api/auth/verify/{token}",
    tag = "认证",
    summary = "验证邮箱",
    description = "根据邮箱验证令牌完成用户邮箱验证。",
    params(
        ("token" = String, Path, description = "邮箱验证令牌")
    ),
    responses(
        (status = 200, description = "邮箱验证成功"),
        (status = 401, description = "验证令牌无效")
    )
)]
#[debug_handler]
async fn verify(State(ctx): State<AppContext>, Path(token): Path<String>) -> Result<Response> {
    let Ok(user) = users::Model::find_by_verification_token(&ctx.db, &token).await else {
        return unauthorized("invalid token");
    };

    if user.email_verified_at.is_some() {
        tracing::info!(pid = user.pid.to_string(), "user already verified");
    } else {
        let active_model = user.into_active_model();
        let user = active_model.verified(&ctx.db).await?;
        tracing::info!(pid = user.pid.to_string(), "user verified");
    }

    format::json(())
}

/// In case the user forgot his password  this endpoints generate a forgot token
/// and send email to the user. In case the email not found in our DB, we are
/// returning a valid request for for security reasons (not exposing users DB
/// list).
#[utoipa::path(
    post,
    path = "/api/auth/forgot",
    tag = "认证",
    summary = "申请重置密码",
    description = "根据邮箱生成重置密码令牌并发送重置邮件；无论邮箱是否存在都返回成功以避免泄露用户信息。",
    request_body = ForgotParams,
    responses(
        (status = 200, description = "重置密码申请已接收")
    )
)]
#[debug_handler]
async fn forgot(
    State(ctx): State<AppContext>,
    Json(params): Json<ForgotParams>,
) -> Result<Response> {
    let Ok(user) = users::Model::find_by_email(&ctx.db, &params.email).await else {
        // we don't want to expose our users email. if the email is invalid we still
        // returning success to the caller
        return format::json(());
    };

    let user = user
        .into_active_model()
        .set_forgot_password_sent(&ctx.db)
        .await?;

    AuthMailer::forgot_password(&ctx, &user).await?;

    format::json(())
}

/// reset user password by the given parameters
#[utoipa::path(
    post,
    path = "/api/auth/reset",
    tag = "认证",
    summary = "重置密码",
    description = "使用重置密码令牌设置新密码。",
    request_body = ResetParams,
    responses(
        (status = 200, description = "重置密码请求已接收")
    )
)]
#[debug_handler]
async fn reset(State(ctx): State<AppContext>, Json(params): Json<ResetParams>) -> Result<Response> {
    let Ok(user) = users::Model::find_by_reset_token(&ctx.db, &params.token).await else {
        // we don't want to expose our users email. if the email is invalid we still
        // returning success to the caller
        tracing::info!("reset token not found");

        return format::json(());
    };
    user.into_active_model()
        .reset_password(&ctx.db, &params.password)
        .await?;

    format::json(())
}

/// Creates a user login and returns a token
#[utoipa::path(
    post,
    path = "/api/auth/login",
    tag = "认证",
    summary = "用户登录",
    description = "使用邮箱和密码登录，成功后返回 JWT。",
    request_body = LoginParams,
    responses(
        (status = 200, description = "登录成功", body = LoginResponse),
        (status = 401, description = "邮箱或密码错误")
    )
)]
#[debug_handler]
async fn login(State(ctx): State<AppContext>, Json(params): Json<LoginParams>) -> Result<Response> {
    let Ok(user) = users::Model::find_by_email(&ctx.db, &params.email).await else {
        tracing::debug!(
            email = params.email,
            "login attempt with non-existent email"
        );
        return unauthorized("Invalid credentials!");
    };

    let valid = user.verify_password(&params.password);

    if !valid {
        return unauthorized("unauthorized!");
    }

    let jwt_secret = ctx.config.get_jwt_config()?;

    let token = user
        .generate_jwt(&jwt_secret.secret, jwt_secret.expiration)
        .or_else(|_| unauthorized("unauthorized!"))?;

    format::json(LoginResponse::new(&user, &token))
}

#[utoipa::path(
    get,
    path = "/api/auth/current",
    tag = "认证",
    summary = "获取当前用户",
    description = "根据 JWT 获取当前登录用户信息。",
    responses(
        (status = 200, description = "当前用户信息", body = CurrentResponse),
        (status = 401, description = "未认证")
    ),
    security(
        ("jwt_token" = [])
    )
)]
#[debug_handler]
async fn current(auth: auth::JWT, State(ctx): State<AppContext>) -> Result<Response> {
    let user = users::Model::find_by_pid(&ctx.db, &auth.claims.pid).await?;
    format::json(CurrentResponse::new(&user))
}

/// Magic link authentication provides a secure and passwordless way to log in to the application.
///
/// # Flow
/// 1. **Request a Magic Link**:
///    A registered user sends a POST request to `/magic-link` with their email.
///    If the email exists, a short-lived, one-time-use token is generated and sent to the user's email.
///    For security and to avoid exposing whether an email exists, the response always returns 200, even if the email is invalid.
///
/// 2. **Click the Magic Link**:
///    The user clicks the link (/magic-link/{token}), which validates the token and its expiration.
///    If valid, the server generates a JWT and responds with a [`LoginResponse`].
///    If invalid or expired, an unauthorized response is returned.
///
/// This flow enhances security by avoiding traditional passwords and providing a seamless login experience.
#[utoipa::path(
    post,
    path = "/api/auth/magic-link",
    tag = "认证",
    summary = "申请魔法链接",
    description = "根据邮箱生成一次性登录链接并发送邮件。",
    request_body = MagicLinkParams,
    responses(
        (status = 200, description = "魔法链接申请已接收"),
        (status = 400, description = "请求无效")
    )
)]
async fn magic_link(
    State(ctx): State<AppContext>,
    Json(params): Json<MagicLinkParams>,
) -> Result<Response> {
    let email_regex = get_allow_email_domain_re();
    if !email_regex.is_match(&params.email) {
        tracing::debug!(
            email = params.email,
            "The provided email is invalid or does not match the allowed domains"
        );
        return bad_request("invalid request");
    }

    let Ok(user) = users::Model::find_by_email(&ctx.db, &params.email).await else {
        // we don't want to expose our users email. if the email is invalid we still
        // returning success to the caller
        tracing::debug!(email = params.email, "user not found by email");
        return format::empty_json();
    };

    let user = user.into_active_model().create_magic_link(&ctx.db).await?;
    AuthMailer::send_magic_link(&ctx, &user).await?;

    format::empty_json()
}

/// Verifies a magic link token and authenticates the user.
#[utoipa::path(
    get,
    path = "/api/auth/magic-link/{token}",
    tag = "认证",
    summary = "验证魔法链接",
    description = "验证一次性魔法链接令牌，成功后返回 JWT。",
    params(
        ("token" = String, Path, description = "魔法链接令牌")
    ),
    responses(
        (status = 200, description = "魔法链接验证成功", body = LoginResponse),
        (status = 401, description = "未认证")
    )
)]
async fn magic_link_verify(
    Path(token): Path<String>,
    State(ctx): State<AppContext>,
) -> Result<Response> {
    let Ok(user) = users::Model::find_by_magic_token(&ctx.db, &token).await else {
        // we don't want to expose our users email. if the email is invalid we still
        // returning success to the caller
        return unauthorized("unauthorized!");
    };

    let user = user.into_active_model().clear_magic_link(&ctx.db).await?;

    let jwt_secret = ctx.config.get_jwt_config()?;

    let token = user
        .generate_jwt(&jwt_secret.secret, jwt_secret.expiration)
        .or_else(|_| unauthorized("unauthorized!"))?;

    format::json(LoginResponse::new(&user, &token))
}

#[utoipa::path(
    post,
    path = "/api/auth/resend-verification-mail",
    tag = "认证",
    summary = "重新发送验证邮件",
    description = "为未完成邮箱验证的用户重新发送验证邮件。",
    request_body = ResendVerificationParams,
    responses(
        (status = 200, description = "重新发送验证邮件请求已接收")
    )
)]
#[debug_handler]
async fn resend_verification_email(
    State(ctx): State<AppContext>,
    Json(params): Json<ResendVerificationParams>,
) -> Result<Response> {
    let Ok(user) = users::Model::find_by_email(&ctx.db, &params.email).await else {
        tracing::info!(
            email = params.email,
            "User not found for resend verification"
        );
        return format::json(());
    };

    if user.email_verified_at.is_some() {
        tracing::info!(
            pid = user.pid.to_string(),
            "User already verified, skipping resend"
        );
        return format::json(());
    }

    let user = user
        .into_active_model()
        .set_email_verification_sent(&ctx.db)
        .await?;

    AuthMailer::send_welcome(&ctx, &user).await?;
    tracing::info!(pid = user.pid.to_string(), "Verification email re-sent");

    format::json(())
}

pub fn routes() -> Routes {
    Routes::new()
        .prefix("/api/auth")
        .add("/register", openapi(post(register), routes!(register)))
        .add("/verify/{token}", openapi(get(verify), routes!(verify)))
        .add("/login", openapi(post(login), routes!(login)))
        .add("/forgot", openapi(post(forgot), routes!(forgot)))
        .add("/reset", openapi(post(reset), routes!(reset)))
        .add("/current", openapi(get(current), routes!(current)))
        .add(
            "/magic-link",
            openapi(post(magic_link), routes!(magic_link)),
        )
        .add(
            "/magic-link/{token}",
            openapi(get(magic_link_verify), routes!(magic_link_verify)),
        )
        .add(
            "/resend-verification-mail",
            openapi(
                post(resend_verification_email),
                routes!(resend_verification_email),
            ),
        )
}
