#version 450

layout(push_constant) uniform FirePushConstants {
    mat4 viewProjection;
    vec4 positionAndSize;
    vec4 cameraRightAndLife;
    vec4 cameraUpAndIntensity;
} pc;

layout(location = 0) out vec2 outUv;

const vec2 kQuadOffsets[6] = vec2[](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main() {
    const vec2 quadOffset = kQuadOffsets[gl_VertexIndex];
    const float horizontalScale = mix(0.45, 0.75, pc.cameraUpAndIntensity.w);
    const float verticalScale = mix(0.9, 1.5, pc.cameraRightAndLife.w);

    const vec3 billboardOffset =
        pc.cameraRightAndLife.xyz * (quadOffset.x * pc.positionAndSize.w * horizontalScale) +
        pc.cameraUpAndIntensity.xyz * (quadOffset.y * pc.positionAndSize.w * verticalScale);

    const vec3 worldPosition = pc.positionAndSize.xyz + billboardOffset;

    gl_Position = pc.viewProjection * vec4(worldPosition, 1.0);
    outUv = quadOffset;
}
