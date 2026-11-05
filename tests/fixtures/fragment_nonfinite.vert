#version 450
layout(location=0) flat out vec4 test0;
layout(location=1) flat out vec4 test1;
layout(push_constant) uniform Controls { uint draw_id; uint capacity; uvec2 pad; uvec4 a; uvec4 b; } pc;
void main() {
    vec2 p[3] = vec2[3](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
    gl_Position = vec4(p[gl_VertexIndex], 0, 1);
    test0 = uintBitsToFloat(pc.a);
    test1 = uintBitsToFloat(pc.b);
}
