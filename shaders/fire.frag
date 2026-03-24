#version 450

layout(push_constant) uniform FirePushConstants {
    mat4 viewProjection;
    vec4 positionAndSize;
    vec4 cameraRightAndLife;
    vec4 cameraUpAndIntensity;
} pc;

layout(location = 0) in vec2 inUv;

layout(location = 0) out vec4 outFragColor;

void main() {
    const float lifeRemaining = clamp(pc.cameraRightAndLife.w, 0.0, 1.0);
    const float intensity = clamp(pc.cameraUpAndIntensity.w, 0.0, 1.0);
    const float height01 = clamp((inUv.y + 1.0) * 0.5, 0.0, 1.0);

    const float taper = mix(1.0, 0.28, height01);
    const float radial = length(vec2(inUv.x / max(taper, 0.001), inUv.y * 0.78));

    const float flameMask = 1.0 - smoothstep(0.18, 1.05, radial);
    const float coreMask = 1.0 - smoothstep(0.0, 0.42, radial);
    const float topFade = 1.0 - smoothstep(0.65, 1.0, height01);
    const float baseShape = 0.55 + 0.45 * smoothstep(0.0, 0.18, height01);

    float glow = flameMask * baseShape * topFade;
    glow *= mix(0.7, 1.15, intensity);
    glow *= lifeRemaining;
    glow += coreMask * (0.35 + intensity * 0.45) * lifeRemaining;

    if (glow < 0.01) {
        discard;
    }

    const vec3 emberRed = vec3(1.0, 0.06, 0.01);
    const vec3 hotRed = vec3(1.0, 0.24, 0.04);
    const vec3 orangeRed = vec3(1.0, 0.48, 0.08);

    vec3 color = mix(emberRed, hotRed, height01 * 0.65 + intensity * 0.2);
    color = mix(color, orangeRed, coreMask * 0.75);

    glow *= 1.1 + coreMask * 0.8;
    outFragColor = vec4(color * glow, glow);
}
