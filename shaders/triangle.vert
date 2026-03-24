#version 450

// The C++ side sends one push-constant block per draw call. That lets us draw
// several objects with different transforms and colors without setting up
// descriptor sets or uniform buffers yet.
layout(push_constant) uniform PushConstants {
    mat4 viewProjection;
    vec4 color;
    vec4 transform;
    ivec4 params;
} pc;

// Pass data to the fragment shader for simple lighting.
layout(location = 0) out vec3 outColor;
layout(location = 1) out vec3 outNormal;

const int kMeshCube = 0;
const int kMeshPlane = 1;

// A cube made from 12 triangles = 36 vertices.
const vec3 kCubePositions[36] = vec3[](
    vec3(-0.5, -0.5,  0.5), vec3( 0.5, -0.5,  0.5), vec3( 0.5,  0.5,  0.5),
    vec3(-0.5, -0.5,  0.5), vec3( 0.5,  0.5,  0.5), vec3(-0.5,  0.5,  0.5),

    vec3( 0.5, -0.5, -0.5), vec3(-0.5, -0.5, -0.5), vec3(-0.5,  0.5, -0.5),
    vec3( 0.5, -0.5, -0.5), vec3(-0.5,  0.5, -0.5), vec3( 0.5,  0.5, -0.5),

    vec3(-0.5, -0.5, -0.5), vec3(-0.5, -0.5,  0.5), vec3(-0.5,  0.5,  0.5),
    vec3(-0.5, -0.5, -0.5), vec3(-0.5,  0.5,  0.5), vec3(-0.5,  0.5, -0.5),

    vec3( 0.5, -0.5,  0.5), vec3( 0.5, -0.5, -0.5), vec3( 0.5,  0.5, -0.5),
    vec3( 0.5, -0.5,  0.5), vec3( 0.5,  0.5, -0.5), vec3( 0.5,  0.5,  0.5),

    vec3(-0.5,  0.5,  0.5), vec3( 0.5,  0.5,  0.5), vec3( 0.5,  0.5, -0.5),
    vec3(-0.5,  0.5,  0.5), vec3( 0.5,  0.5, -0.5), vec3(-0.5,  0.5, -0.5),

    vec3(-0.5, -0.5, -0.5), vec3( 0.5, -0.5, -0.5), vec3( 0.5, -0.5,  0.5),
    vec3(-0.5, -0.5, -0.5), vec3( 0.5, -0.5,  0.5), vec3(-0.5, -0.5,  0.5)
);

const vec3 kCubeNormals[36] = vec3[](
    vec3( 0.0,  0.0,  1.0), vec3( 0.0,  0.0,  1.0), vec3( 0.0,  0.0,  1.0),
    vec3( 0.0,  0.0,  1.0), vec3( 0.0,  0.0,  1.0), vec3( 0.0,  0.0,  1.0),

    vec3( 0.0,  0.0, -1.0), vec3( 0.0,  0.0, -1.0), vec3( 0.0,  0.0, -1.0),
    vec3( 0.0,  0.0, -1.0), vec3( 0.0,  0.0, -1.0), vec3( 0.0,  0.0, -1.0),

    vec3(-1.0,  0.0,  0.0), vec3(-1.0,  0.0,  0.0), vec3(-1.0,  0.0,  0.0),
    vec3(-1.0,  0.0,  0.0), vec3(-1.0,  0.0,  0.0), vec3(-1.0,  0.0,  0.0),

    vec3( 1.0,  0.0,  0.0), vec3( 1.0,  0.0,  0.0), vec3( 1.0,  0.0,  0.0),
    vec3( 1.0,  0.0,  0.0), vec3( 1.0,  0.0,  0.0), vec3( 1.0,  0.0,  0.0),

    vec3( 0.0,  1.0,  0.0), vec3( 0.0,  1.0,  0.0), vec3( 0.0,  1.0,  0.0),
    vec3( 0.0,  1.0,  0.0), vec3( 0.0,  1.0,  0.0), vec3( 0.0,  1.0,  0.0),

    vec3( 0.0, -1.0,  0.0), vec3( 0.0, -1.0,  0.0), vec3( 0.0, -1.0,  0.0),
    vec3( 0.0, -1.0,  0.0), vec3( 0.0, -1.0,  0.0), vec3( 0.0, -1.0,  0.0)
);

// A square plane made from two triangles.
const vec3 kPlanePositions[6] = vec3[](
    vec3(-0.5, 0.0, -0.5), vec3( 0.5, 0.0, -0.5), vec3( 0.5, 0.0,  0.5),
    vec3(-0.5, 0.0, -0.5), vec3( 0.5, 0.0,  0.5), vec3(-0.5, 0.0,  0.5)
);

const vec3 kPlaneNormals[6] = vec3[](
    vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0),
    vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0)
);

void main() {
    const bool isCube = pc.params.x == kMeshCube;
    const vec3 localPosition = isCube ? kCubePositions[gl_VertexIndex] : kPlanePositions[gl_VertexIndex];
    const vec3 localNormal = isCube ? kCubeNormals[gl_VertexIndex] : kPlaneNormals[gl_VertexIndex];

    // transform.xyz = world-space position, transform.w = uniform scale.
    const vec3 worldPosition = localPosition * pc.transform.w + pc.transform.xyz;

    gl_Position = pc.viewProjection * vec4(worldPosition, 1.0);
    outColor = pc.color.rgb;
    outNormal = localNormal;
}
