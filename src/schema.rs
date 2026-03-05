//! Schema system for ForgeEngine
//!
//! Design parameter parsing, schema resolution, buffer validation,
//! and test buffer generation. Invariant evaluation via embedded QuickJS.

use indexmap::IndexMap;
use rand::Rng;
use rquickjs::{Context, Function, Object, Runtime};
use serde::Deserialize;
use std::collections::HashMap;

// ===== fix16 helpers =====

pub fn to_fix16(value: f64) -> i64 {
    (value * 65536.0).round() as i64
}

pub fn from_fix16(value: i64) -> f64 {
    value as f64 / 65536.0
}

/// Fixed-point 16.16 multiply. Rust has i64, so we can do this directly.
/// Matches the WGSL split-half approach in result (but simpler implementation).
pub fn fix16_mul(a: i64, b: i64) -> i64 {
    (a * b) >> 16
}

// ===== Design Parameters =====

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParamType {
    Fix16,
    U32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ParamDef {
    #[serde(rename = "type")]
    pub type_: ParamType,
    pub value: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DesignParamsJSON {
    pub name: String,
    pub params: IndexMap<String, ParamDef>,
    pub design_invariants: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ResolvedParams {
    pub name: String,
    pub raw: HashMap<String, i64>,
    pub display: HashMap<String, f64>,
    pub types: HashMap<String, ParamType>,
}

pub fn resolve_params(json: &DesignParamsJSON) -> Result<ResolvedParams, String> {
    let mut raw = HashMap::new();
    let mut display = HashMap::new();
    let mut types = HashMap::new();

    for (name, def) in &json.params {
        match def.type_ {
            ParamType::Fix16 => {
                raw.insert(name.clone(), to_fix16(def.value));
                display.insert(name.clone(), def.value);
                types.insert(name.clone(), ParamType::Fix16);
            }
            ParamType::U32 => {
                raw.insert(name.clone(), def.value as i64);
                display.insert(name.clone(), def.value);
                types.insert(name.clone(), ParamType::U32);
            }
        }
    }

    Ok(ResolvedParams {
        name: json.name.clone(),
        raw,
        display,
        types,
    })
}

// ===== Invariant Checking via QuickJS =====

pub struct InvariantResult {
    pub valid: bool,
    pub violations: Vec<String>,
}

/// Set up fix16_mul as a JS global function in the given QuickJS context.
fn setup_fix16_mul(ctx: rquickjs::Ctx<'_>) -> rquickjs::Result<()> {
    ctx.globals().set(
        "fix16_mul",
        Function::new(ctx.clone(), |a: f64, b: f64| -> f64 {
            let a = a as i64;
            let b = b as i64;
            ((a * b) >> 16) as f64
        })?,
    )
}

/// Set the `$` global object with raw param values.
fn setup_dollar(ctx: rquickjs::Ctx<'_>, params: &HashMap<String, i64>) -> rquickjs::Result<()> {
    let dollar = Object::new(ctx.clone())?;
    for (k, v) in params {
        dollar.set(k.as_str(), *v as f64)?;
    }
    ctx.globals().set("$", dollar)
}

/// Evaluate a JS expression and return whether it's truthy.
fn eval_truthy(ctx: &rquickjs::Ctx<'_>, expr: &str) -> Result<bool, String> {
    let script = format!("!!({})", expr);
    match ctx.eval::<rquickjs::Value, String>(script) {
        Ok(val) => Ok(val.as_bool().unwrap_or_else(|| {
            // Fallback: check numeric truthiness
            val.as_int().map(|n| n != 0).unwrap_or(false)
        })),
        Err(e) => Err(format!("{}", e)),
    }
}

pub fn check_design_invariants(
    params: &ResolvedParams,
    invariants: &[String],
) -> InvariantResult {
    if invariants.is_empty() {
        return InvariantResult {
            valid: true,
            violations: vec![],
        };
    }

    let rt = Runtime::new().expect("Failed to create QuickJS runtime");
    let ctx = Context::full(&rt).expect("Failed to create QuickJS context");
    let mut violations = Vec::new();

    ctx.with(|ctx| {
        setup_fix16_mul(ctx.clone()).expect("Failed to setup fix16_mul");
        setup_dollar(ctx.clone(), &params.raw).expect("Failed to setup $ object");

        for expr in invariants {
            match eval_truthy(&ctx, expr) {
                Ok(true) => {}
                Ok(false) => violations.push(format!("Invariant violated: {}", expr)),
                Err(e) => violations.push(format!("Invariant error: {} — {}", expr, e)),
            }
        }
    });

    InvariantResult {
        valid: violations.is_empty(),
        violations,
    }
}

// ===== Buffer Schema =====

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldType {
    U32,
    I32,
    Fix16,
    Fix16x2,
    Vec2f,
    Vec3f,
    Vec4f,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BufferCategory {
    State,
    Transient,
    Input,
    Output,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FieldDef {
    #[serde(rename = "type")]
    pub type_: FieldType,
    #[serde(default)]
    pub range: Option<serde_json::Value>,
    #[serde(rename = "enum", default)]
    pub enum_: Option<Vec<i64>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BufferSchemaJSON {
    pub name: String,
    #[serde(rename = "struct")]
    pub struct_: IndexMap<String, FieldDef>,
    pub capacity: serde_json::Value,
    pub buffer_category: BufferCategory,
    #[serde(default)]
    pub invariants: Vec<String>,
}

// ===== Resolved Schema Types =====

#[derive(Debug, Clone)]
pub enum ResolvedRange {
    Simple(i64, i64),
    Nested((i64, i64), (i64, i64)),
}

#[derive(Debug, Clone)]
pub struct ResolvedFieldDef {
    pub name: String,
    pub type_: FieldType,
    pub byte_offset: usize,
    pub byte_size: usize,
    pub range: Option<ResolvedRange>,
    pub enum_values: Option<Vec<i64>>,
}

#[derive(Debug, Clone)]
pub struct ResolvedSchema {
    pub name: String,
    pub fields: Vec<ResolvedFieldDef>,
    pub struct_size: usize,
    pub capacity: usize,
    pub buffer_category: BufferCategory,
    pub invariant_sources: Vec<String>,
    pub total_size: usize,
}

// Field size and alignment (matching WGSL layout rules)
fn field_size(ft: FieldType) -> usize {
    match ft {
        FieldType::U32 | FieldType::I32 | FieldType::Fix16 => 4,
        FieldType::Fix16x2 | FieldType::Vec2f => 8,
        FieldType::Vec3f => 12,
        FieldType::Vec4f => 16,
    }
}

fn field_alignment(ft: FieldType) -> usize {
    match ft {
        FieldType::U32 | FieldType::I32 | FieldType::Fix16 | FieldType::Fix16x2 => 4,
        FieldType::Vec2f => 8,
        FieldType::Vec3f | FieldType::Vec4f => 16,
    }
}

fn is_float_type(ft: FieldType) -> bool {
    matches!(ft, FieldType::Vec2f | FieldType::Vec3f | FieldType::Vec4f)
}

// ===== $-reference resolution =====

pub fn resolve_ref(val: &serde_json::Value, params: &ResolvedParams) -> Result<i64, String> {
    match val {
        serde_json::Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f as i64))
            .ok_or_else(|| format!("Invalid number: {}", n)),
        serde_json::Value::String(s) => {
            if s.contains('$') {
                return resolve_ref_expr(s, params);
            }
            s.parse::<i64>()
                .map_err(|_| format!("Invalid ref: {}", s))
        }
        _ => Err(format!("Invalid ref value: {:?}", val)),
    }
}

fn resolve_ref_expr(expr: &str, params: &ResolvedParams) -> Result<i64, String> {
    // Substitute $-param (negated) before $param to avoid partial matches
    // Uses raw values (fix16-encoded) for correct range/value resolution
    let mut js_expr = expr.to_string();
    for (name, val) in &params.raw {
        let neg_pattern = format!("$-{}", name);
        js_expr = js_expr.replace(&neg_pattern, &format!("({})", -val));
    }
    for (name, val) in &params.raw {
        let pattern = format!("${}", name);
        js_expr = js_expr.replace(&pattern, &val.to_string());
    }
    // Check for unresolved references
    if js_expr.contains('$') {
        return Err(format!("Unresolved param reference in expression: {}", expr));
    }
    let rt = rquickjs::Runtime::new().map_err(|e| format!("QuickJS: {}", e))?;
    let ctx = rquickjs::Context::full(&rt).map_err(|e| format!("QuickJS: {}", e))?;
    ctx.with(|ctx| {
        let val: f64 = ctx
            .eval(js_expr.as_bytes())
            .map_err(|e| format!("Expression '{}' eval failed: {}", expr, e))?;
        Ok(val as i64)
    })
}

fn resolve_range(
    range: &Option<serde_json::Value>,
    params: &ResolvedParams,
) -> Result<Option<ResolvedRange>, String> {
    let range = match range {
        Some(v) => v,
        None => return Ok(None),
    };

    let arr = range
        .as_array()
        .ok_or_else(|| "Range must be an array".to_string())?;
    if arr.len() != 2 {
        return Err("Range must have exactly 2 elements".into());
    }

    // Check if nested: [[min, max], [min, max]]
    if arr[0].is_array() && arr[1].is_array() {
        let inner0 = arr[0].as_array().unwrap();
        let inner1 = arr[1].as_array().unwrap();
        if inner0.len() != 2 || inner1.len() != 2 {
            return Err("Nested range components must have 2 elements each".into());
        }
        Ok(Some(ResolvedRange::Nested(
            (
                resolve_ref(&inner0[0], params)?,
                resolve_ref(&inner0[1], params)?,
            ),
            (
                resolve_ref(&inner1[0], params)?,
                resolve_ref(&inner1[1], params)?,
            ),
        )))
    } else {
        Ok(Some(ResolvedRange::Simple(
            resolve_ref(&arr[0], params)?,
            resolve_ref(&arr[1], params)?,
        )))
    }
}

// ===== Schema Resolution =====

pub fn resolve_schema(
    json: &BufferSchemaJSON,
    params: &ResolvedParams,
) -> Result<ResolvedSchema, String> {
    // Float-in-state constraint
    if json.buffer_category == BufferCategory::State {
        for (name, field) in &json.struct_ {
            if is_float_type(field.type_) {
                return Err(format!(
                    "Float type \"{:?}\" not allowed in state buffer field \"{}\"",
                    field.type_, name
                ));
            }
        }
    }

    // Build raw field list, expanding fix16x2
    struct RawField {
        name: String,
        type_: FieldType,
        range: Option<ResolvedRange>,
        enum_values: Option<Vec<i64>>,
    }

    let mut raw_fields = Vec::new();

    for (name, field) in &json.struct_ {
        let resolved_range = resolve_range(&field.range, params)?;

        if field.type_ == FieldType::Fix16x2 {
            // Expand to two i32 fields
            let (range_x, range_y) = match resolved_range {
                Some(ResolvedRange::Nested(x, y)) => (
                    Some(ResolvedRange::Simple(x.0, x.1)),
                    Some(ResolvedRange::Simple(y.0, y.1)),
                ),
                None => (None, None),
                Some(ResolvedRange::Simple(..)) => {
                    return Err(format!(
                        "fix16x2 field '{}' requires nested range [[minX, maxX], [minY, maxY]]",
                        name
                    ));
                }
            };
            raw_fields.push(RawField {
                name: format!("{}_x", name),
                type_: FieldType::I32,
                range: range_x,
                enum_values: None,
            });
            raw_fields.push(RawField {
                name: format!("{}_y", name),
                type_: FieldType::I32,
                range: range_y,
                enum_values: None,
            });
        } else {
            raw_fields.push(RawField {
                name: name.clone(),
                type_: field.type_,
                range: resolved_range,
                enum_values: field.enum_.clone(),
            });
        }
    }

    // Compute struct layout with alignment
    let mut fields = Vec::new();
    let mut offset = 0usize;

    for rf in &raw_fields {
        let align = field_alignment(rf.type_);
        let size = field_size(rf.type_);

        // Align offset
        if offset % align != 0 {
            offset += align - (offset % align);
        }

        fields.push(ResolvedFieldDef {
            name: rf.name.clone(),
            type_: rf.type_,
            byte_offset: offset,
            byte_size: size,
            range: rf.range.clone(),
            enum_values: rf.enum_values.clone(),
        });

        offset += size;
    }

    // Pad struct to 4-byte boundary
    if offset % 4 != 0 {
        offset += 4 - (offset % 4);
    }
    let struct_size = offset;

    // Resolve capacity
    let capacity = resolve_ref(&json.capacity, params)? as usize;

    Ok(ResolvedSchema {
        name: json.name.clone(),
        fields,
        struct_size,
        capacity,
        buffer_category: json.buffer_category,
        invariant_sources: json.invariants.clone(),
        total_size: struct_size * capacity,
    })
}

// ===== Buffer I/O =====

pub fn read_field(buf: &[u8], offset: usize, ft: FieldType) -> i64 {
    match ft {
        FieldType::U32 => u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as i64,
        FieldType::I32 | FieldType::Fix16 => {
            i32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap()) as i64
        }
        FieldType::Fix16x2 | FieldType::Vec2f | FieldType::Vec3f | FieldType::Vec4f => {
            unreachable!("compound types should be expanded before reading")
        }
    }
}

pub fn write_field(buf: &mut [u8], offset: usize, ft: FieldType, value: i64) {
    match ft {
        FieldType::U32 => {
            buf[offset..offset + 4].copy_from_slice(&(value as u32).to_le_bytes());
        }
        FieldType::I32 | FieldType::Fix16 => {
            buf[offset..offset + 4].copy_from_slice(&(value as i32).to_le_bytes());
        }
        FieldType::Fix16x2 | FieldType::Vec2f | FieldType::Vec3f | FieldType::Vec4f => {
            unreachable!("compound types should be expanded before writing")
        }
    }
}

pub fn read_element(buf: &[u8], schema: &ResolvedSchema, index: usize) -> HashMap<String, i64> {
    let base = index * schema.struct_size;
    let mut result = HashMap::new();
    for field in &schema.fields {
        let offset = base + field.byte_offset;
        result.insert(field.name.clone(), read_field(buf, offset, field.type_));
    }
    result
}

pub fn write_element(
    buf: &mut [u8],
    schema: &ResolvedSchema,
    index: usize,
    values: &HashMap<String, i64>,
) {
    let base = index * schema.struct_size;
    for field in &schema.fields {
        let offset = base + field.byte_offset;
        let value = values.get(&field.name).copied().unwrap_or(0);
        write_field(buf, offset, field.type_, value);
    }
}

// ===== Validation =====

pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
}

const MAX_ERRORS: usize = 20;

pub fn validate_buffer(
    buf: &[u8],
    schema: &ResolvedSchema,
    params: &ResolvedParams,
) -> ValidationResult {
    let mut errors = Vec::new();

    // Level 1: byte length
    if buf.len() != schema.total_size {
        return ValidationResult {
            valid: false,
            errors: vec![format!(
                "Buffer length mismatch: expected {}, got {}",
                schema.total_size,
                buf.len()
            )],
        };
    }

    // Set up QuickJS once if we have invariants
    let has_invariants = !schema.invariant_sources.is_empty();
    let rt_holder = if has_invariants {
        Some(Runtime::new().expect("Failed to create QuickJS runtime"))
    } else {
        None
    };
    let ctx_holder = rt_holder
        .as_ref()
        .map(|rt| Context::full(rt).expect("QuickJS context"));

    for i in 0..schema.capacity {
        if errors.len() >= MAX_ERRORS {
            break;
        }
        let element = read_element(buf, schema, i);

        // Level 2: per-field range and enum checks
        for field in &schema.fields {
            if errors.len() >= MAX_ERRORS {
                break;
            }
            let value = element[&field.name];

            if let Some(ref enum_values) = field.enum_values {
                if !enum_values.contains(&value) {
                    errors.push(format!(
                        "Element {}, field \"{}\": value {} not in enum {:?}",
                        i, field.name, value, enum_values
                    ));
                }
            }

            if let Some(ref range) = field.range {
                if let ResolvedRange::Simple(min, max) = range {
                    if value < *min || value > *max {
                        errors.push(format!(
                            "Element {}, field \"{}\": value {} out of range [{}, {}]",
                            i, field.name, value, min, max
                        ));
                    }
                }
            }
        }

        // Level 3: per-element invariants via QuickJS
        if has_invariants && errors.len() < MAX_ERRORS {
            let ctx_ref = ctx_holder.as_ref().unwrap();
            ctx_ref.with(|ctx| {
                let globals = ctx.globals();

                // Set field values as globals
                for (name, val) in &element {
                    let _ = globals.set(name.as_str(), *val as f64);
                }

                // Set $ and fix16_mul
                let _ = setup_dollar(ctx.clone(), &params.raw);
                let _ = setup_fix16_mul(ctx.clone());

                for expr in &schema.invariant_sources {
                    if errors.len() >= MAX_ERRORS {
                        break;
                    }
                    match eval_truthy(&ctx, expr) {
                        Ok(true) => {}
                        Ok(false) => {
                            errors.push(format!(
                                "Element {}: invariant violated: {}",
                                i, expr
                            ));
                        }
                        Err(e) => {
                            errors.push(format!(
                                "Element {}: invariant error: {} — {}",
                                i, expr, e
                            ));
                        }
                    }
                }
            });
        }
    }

    ValidationResult {
        valid: errors.is_empty(),
        errors,
    }
}

// ===== Buffer Generation =====

fn random_field_value(field: &ResolvedFieldDef, rng: &mut impl Rng) -> i64 {
    if let Some(ref enum_values) = field.enum_values {
        return enum_values[rng.gen_range(0..enum_values.len())];
    }
    if let Some(ResolvedRange::Simple(min, max)) = &field.range {
        if min == max {
            return *min;
        }
        return rng.gen_range(*min..=*max);
    }
    0
}

fn min_field_value(field: &ResolvedFieldDef) -> i64 {
    if let Some(ref enum_values) = field.enum_values {
        return enum_values[0];
    }
    if let Some(ResolvedRange::Simple(min, _)) = &field.range {
        return *min;
    }
    0
}

fn max_field_value(field: &ResolvedFieldDef) -> i64 {
    if let Some(ref enum_values) = field.enum_values {
        return *enum_values.last().unwrap();
    }
    if let Some(ResolvedRange::Simple(_, max)) = &field.range {
        return *max;
    }
    0
}

fn build_element(
    schema: &ResolvedSchema,
    value_fn: &mut dyn FnMut(&ResolvedFieldDef) -> i64,
) -> HashMap<String, i64> {
    let mut result = HashMap::new();
    for field in &schema.fields {
        result.insert(field.name.clone(), value_fn(field));
    }
    result
}

fn check_element_invariants(
    element: &HashMap<String, i64>,
    schema: &ResolvedSchema,
    params: &ResolvedParams,
) -> bool {
    if schema.invariant_sources.is_empty() {
        return true;
    }

    let rt = Runtime::new().expect("QuickJS runtime");
    let ctx = Context::full(&rt).expect("QuickJS context");
    let mut ok = true;

    ctx.with(|ctx| {
        let globals = ctx.globals();
        for (name, val) in element {
            let _ = globals.set(name.as_str(), *val as f64);
        }
        let _ = setup_dollar(ctx.clone(), &params.raw);
        let _ = setup_fix16_mul(ctx.clone());

        for expr in &schema.invariant_sources {
            match eval_truthy(&ctx, expr) {
                Ok(true) => {}
                _ => {
                    ok = false;
                    return;
                }
            }
        }
    });

    ok
}

pub fn generate_random_element(
    schema: &ResolvedSchema,
    params: &ResolvedParams,
) -> Result<HashMap<String, i64>, String> {
    let mut rng = rand::thread_rng();
    for _ in 0..100 {
        let el = build_element(schema, &mut |field| random_field_value(field, &mut rng));
        if check_element_invariants(&el, schema, params) {
            return Ok(el);
        }
    }
    Err(format!(
        "Failed to generate valid random element for \"{}\" after 100 attempts",
        schema.name
    ))
}

pub fn generate_random_buffer(
    schema: &ResolvedSchema,
    params: &ResolvedParams,
) -> Result<Vec<u8>, String> {
    let mut buf = vec![0u8; schema.total_size];
    for i in 0..schema.capacity {
        let el = generate_random_element(schema, params)?;
        write_element(&mut buf, schema, i, &el);
    }
    Ok(buf)
}

#[derive(Debug, Clone, Copy)]
pub enum AdversarialStrategy {
    Zero,
    Min,
    Max,
    Boundary,
}

pub fn generate_adversarial_element(
    schema: &ResolvedSchema,
    params: &ResolvedParams,
    strategy: AdversarialStrategy,
) -> Result<HashMap<String, i64>, String> {
    let mut rng = rand::thread_rng();
    let el = match strategy {
        AdversarialStrategy::Zero => build_element(schema, &mut |_| 0),
        AdversarialStrategy::Min => build_element(schema, &mut |f| min_field_value(f)),
        AdversarialStrategy::Max => build_element(schema, &mut |f| max_field_value(f)),
        AdversarialStrategy::Boundary => build_element(schema, &mut |field| {
            if rng.gen_bool(0.5) {
                min_field_value(field)
            } else {
                max_field_value(field)
            }
        }),
    };

    if check_element_invariants(&el, schema, params) {
        Ok(el)
    } else {
        generate_random_element(schema, params)
    }
}

pub fn generate_adversarial_buffers(
    schema: &ResolvedSchema,
    params: &ResolvedParams,
) -> Result<Vec<Vec<u8>>, String> {
    let strategies = [
        AdversarialStrategy::Zero,
        AdversarialStrategy::Min,
        AdversarialStrategy::Max,
        AdversarialStrategy::Boundary,
    ];
    let mut buffers = Vec::new();

    for strategy in &strategies {
        let mut buf = vec![0u8; schema.total_size];
        for i in 0..schema.capacity {
            let el = generate_adversarial_element(schema, params, *strategy)?;
            write_element(&mut buf, schema, i, &el);
        }
        buffers.push(buf);
    }

    // Single-active: one random valid element, rest zero
    let mut buf = vec![0u8; schema.total_size];
    let el = generate_random_element(schema, params)?;
    write_element(&mut buf, schema, 0, &el);
    buffers.push(buf);

    Ok(buffers)
}

// ===== Tests =====

#[cfg(test)]
mod tests {
    use super::*;

    fn test_design_params() -> DesignParamsJSON {
        serde_json::from_str(
            r#"{
            "name": "test_design",
            "params": {
                "world_width": { "type": "fix16", "value": 1280 },
                "world_height": { "type": "fix16", "value": 720 },
                "max_speed": { "type": "fix16", "value": 500 },
                "player_health": { "type": "u32", "value": 100 },
                "enemy_health": { "type": "u32", "value": 30 },
                "boss_health": { "type": "u32", "value": 200 },
                "bullet_damage": { "type": "u32", "value": 10 },
                "bullet_health": { "type": "u32", "value": 1 },
                "max_entities": { "type": "u32", "value": 8 },
                "player_size": { "type": "fix16", "value": 16 },
                "max_delta_time": { "type": "fix16", "value": 0.05 }
            },
            "design_invariants": [
                "$.bullet_damage < $.player_health",
                "$.boss_health >= $.enemy_health",
                "$.max_entities >= 8",
                "fix16_mul($.max_speed, $.max_delta_time) < $.world_width"
            ]
        }"#,
        )
        .unwrap()
    }

    fn test_entity_schema() -> BufferSchemaJSON {
        serde_json::from_str(
            r#"{
            "name": "EntityBuffer",
            "struct": {
                "position": {
                    "type": "fix16x2",
                    "range": [[0, "$world_width"], [0, "$world_height"]]
                },
                "health": { "type": "u32", "range": [0, "$boss_health"] },
                "max_health": { "type": "u32", "range": [0, "$boss_health"] },
                "entity_type": { "type": "u32", "enum": [0, 1, 2, 3] },
                "alive": { "type": "u32", "enum": [0, 1] }
            },
            "capacity": "$max_entities",
            "buffer_category": "state",
            "invariants": ["health <= max_health"]
        }"#,
        )
        .unwrap()
    }

    #[test]
    fn param_resolution() {
        let json = test_design_params();
        let params = resolve_params(&json).unwrap();

        // fix16 values shifted
        assert_eq!(params.raw["world_width"], to_fix16(1280.0));
        assert_eq!(params.raw["max_delta_time"], to_fix16(0.05));
        assert_eq!(params.display["world_width"], 1280.0);

        // u32 values as-is
        assert_eq!(params.raw["player_health"], 100);
        assert_eq!(params.raw["max_entities"], 8);
    }

    #[test]
    fn design_invariants_pass() {
        let json = test_design_params();
        let params = resolve_params(&json).unwrap();
        let result = check_design_invariants(&params, &json.design_invariants);
        assert!(result.valid, "Violations: {:?}", result.violations);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn design_invariants_fail() {
        let json: DesignParamsJSON = serde_json::from_str(
            r#"{
            "name": "broken",
            "params": {
                "player_health": { "type": "u32", "value": 5 },
                "bullet_damage": { "type": "u32", "value": 10 }
            },
            "design_invariants": ["$.bullet_damage < $.player_health"]
        }"#,
        )
        .unwrap();

        let params = resolve_params(&json).unwrap();
        let result = check_design_invariants(&params, &json.design_invariants);
        assert!(!result.valid);
        assert!(!result.violations.is_empty());
        assert!(result.violations[0].contains("violated"));
    }

    #[test]
    fn schema_resolution() {
        let design = test_design_params();
        let params = resolve_params(&design).unwrap();
        let schema_json = test_entity_schema();
        let schema = resolve_schema(&schema_json, &params).unwrap();

        // fix16x2 expands to two i32 fields: position_x(4) + position_y(4) = 8
        // health(4) + max_health(4) + entity_type(4) + alive(4) = 16
        // total struct = 24 bytes
        assert_eq!(schema.struct_size, 24);
        assert_eq!(schema.capacity, 8);
        assert_eq!(schema.total_size, 24 * 8);

        // Verify field names after expansion
        let names: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(
            names,
            &["position_x", "position_y", "health", "max_health", "entity_type", "alive"]
        );
    }

    #[test]
    fn ref_resolution() {
        let design = test_design_params();
        let params = resolve_params(&design).unwrap();

        // $world_width resolves to raw fix16 value
        let val: serde_json::Value = serde_json::json!("$world_width");
        assert_eq!(resolve_ref(&val, &params).unwrap(), to_fix16(1280.0));

        // $-max_speed resolves to negated raw value
        let val: serde_json::Value = serde_json::json!("$-max_speed");
        assert_eq!(resolve_ref(&val, &params).unwrap(), -to_fix16(500.0));

        // numeric literal passes through
        let val: serde_json::Value = serde_json::json!(42);
        assert_eq!(resolve_ref(&val, &params).unwrap(), 42);
    }

    #[test]
    fn validate_correct_buffer() {
        let design = test_design_params();
        let params = resolve_params(&design).unwrap();
        let schema = resolve_schema(&test_entity_schema(), &params).unwrap();

        let buf = generate_random_buffer(&schema, &params).unwrap();
        let result = validate_buffer(&buf, &schema, &params);
        assert!(result.valid, "Errors: {:?}", result.errors);
    }

    #[test]
    fn validate_wrong_length() {
        let design = test_design_params();
        let params = resolve_params(&design).unwrap();
        let schema = resolve_schema(&test_entity_schema(), &params).unwrap();

        let buf = vec![0u8; 10];
        let result = validate_buffer(&buf, &schema, &params);
        assert!(!result.valid);
        assert!(result.errors[0].contains("length"));
    }

    #[test]
    fn validate_out_of_range() {
        let design = test_design_params();
        let params = resolve_params(&design).unwrap();
        let schema = resolve_schema(&test_entity_schema(), &params).unwrap();

        let mut buf = vec![0u8; schema.total_size];
        // health field is at byte offset 8, boss_health = 200
        // Write 999 which is > 200
        buf[8..12].copy_from_slice(&999u32.to_le_bytes());
        // max_health also out of range to avoid invariant masking
        buf[12..16].copy_from_slice(&999u32.to_le_bytes());

        let result = validate_buffer(&buf, &schema, &params);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("range")));
    }

    #[test]
    fn validate_bad_enum() {
        let design = test_design_params();
        let params = resolve_params(&design).unwrap();
        let schema = resolve_schema(&test_entity_schema(), &params).unwrap();

        let mut buf = vec![0u8; schema.total_size];
        // entity_type is at byte offset 16, valid values are [0,1,2,3]
        buf[16..20].copy_from_slice(&99u32.to_le_bytes());

        let result = validate_buffer(&buf, &schema, &params);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("enum")));
    }

    #[test]
    fn validate_invariant_violation() {
        let design = test_design_params();
        let params = resolve_params(&design).unwrap();
        let schema = resolve_schema(&test_entity_schema(), &params).unwrap();

        let mut buf = vec![0u8; schema.total_size];
        // Set health=100, max_health=50 → violates "health <= max_health"
        buf[8..12].copy_from_slice(&100u32.to_le_bytes());
        buf[12..16].copy_from_slice(&50u32.to_le_bytes());
        // Set valid enum values
        buf[16..20].copy_from_slice(&1u32.to_le_bytes());
        buf[20..24].copy_from_slice(&1u32.to_le_bytes());

        let result = validate_buffer(&buf, &schema, &params);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| e.contains("invariant")));
    }

    #[test]
    fn adversarial_generation() {
        let design = test_design_params();
        let params = resolve_params(&design).unwrap();
        let schema = resolve_schema(&test_entity_schema(), &params).unwrap();

        let buffers = generate_adversarial_buffers(&schema, &params).unwrap();
        assert!(buffers.len() >= 4);

        for (i, buf) in buffers.iter().enumerate() {
            assert_eq!(buf.len(), schema.total_size);
            let result = validate_buffer(buf, &schema, &params);
            assert!(
                result.valid,
                "Adversarial buffer {} failed validation: {:?}",
                i, result.errors
            );
        }
    }

    #[test]
    fn float_in_state_rejected() {
        let design = test_design_params();
        let params = resolve_params(&design).unwrap();
        let schema_json: BufferSchemaJSON = serde_json::from_str(
            r#"{
            "name": "BadState",
            "struct": { "pos": { "type": "vec2f" } },
            "capacity": 10,
            "buffer_category": "state"
        }"#,
        )
        .unwrap();

        let err = resolve_schema(&schema_json, &params).unwrap_err();
        assert!(err.to_lowercase().contains("float") || err.to_lowercase().contains("not allowed"));
    }

    #[test]
    fn fix16_mul_correctness() {
        // 1 × 1 = 1
        assert_eq!(fix16_mul(to_fix16(1.0), to_fix16(1.0)), to_fix16(1.0));
        // 2 × 3 = 6
        assert_eq!(fix16_mul(to_fix16(2.0), to_fix16(3.0)), to_fix16(6.0));
        // 0.5 × 0.5 = 0.25
        assert_eq!(fix16_mul(to_fix16(0.5), to_fix16(0.5)), to_fix16(0.25));
        // negatives
        assert_eq!(fix16_mul(to_fix16(-2.0), to_fix16(3.0)), to_fix16(-6.0));
        // max_speed × max_delta_time ≈ 25
        let result = fix16_mul(to_fix16(500.0), to_fix16(0.05));
        assert!((from_fix16(result) - 25.0).abs() < 0.01);
    }
}
