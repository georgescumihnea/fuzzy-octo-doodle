#version 450

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec4 outFragColor;

void main() {
    const vec3 normal = normalize(inNormal);

    // A fixed sunlight direction is enough to make the cubes read as 3D.
    const vec3 lightDirection = normalize(vec3(-0.6, 0.8, -0.3));
    const float diffuse = max(dot(normal, lightDirection), 0.0);
    const float ambient = 0.25;
    const vec3 litColor = inColor * (ambient + diffuse * 0.75);

    outFragColor = vec4(litColor, 1.0);
}
