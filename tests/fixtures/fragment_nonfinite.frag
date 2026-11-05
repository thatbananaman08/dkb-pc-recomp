#version 450
// Standalone test fixture: this layout is not the renderer's descriptor ABI.
layout(location = 0) out vec4 color0;
layout(location = 1) out vec4 color1;
layout(location = 0) flat in vec4 test0;
layout(location = 1) flat in vec4 test1;
layout(push_constant) uniform Controls { uint draw_id; uint capacity; } controls;
#ifdef DIAGNOSTIC
#ifndef REPORT_SET
#define REPORT_SET 1
#endif
layout(set = REPORT_SET, binding = 0, std430) buffer Report { uint flags[]; } report;
void note_invalid(vec4 v, uint output_location) {
    uint flags = (any(isnan(v)) ? 1u : 0u) | (any(isinf(v)) ? 2u : 0u);
    if (flags != 0u && controls.draw_id < controls.capacity)
        atomicOr(report.flags[controls.draw_id], flags << (2u * output_location));
}
#endif
void main() {
    // Values are supplied at runtime so the compiler cannot fold away non-finite cases.
    vec4 a = test0;
    vec4 b = test1;
#ifdef SECOND_SHADER
    if (controls.draw_id == 0xffffffffu) a = -a;
#endif
    color0.x = a.x;
    color0.y = a.y;
    color0.z = a.z;
    color0.w = a.w;
    // Draw 4 deliberately leaves this component unwritten on both variants.
    if (controls.draw_id != 4u) color1.x = b.x;
    color1.y = b.y;
    color1.z = b.z;
    color1.w = b.w;
#ifdef DIAGNOSTIC
    note_invalid(a, 0u);
    note_invalid(vec4(controls.draw_id != 4u ? b.x : 0.0, b.yzw), 1u);
#endif
}

