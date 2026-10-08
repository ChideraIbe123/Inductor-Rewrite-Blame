from rewrite_blame.codegen_stats import analyze_source, analyze_sources, diff_stats

CPP_SRC = '''
async_compile = AsyncCompile()

cpp_fused_add_mul_0 = async_compile.cpp_pybinding(['const float*', 'const float*', 'float*'], r\'\'\'
extern "C"  void  kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    #pragma omp parallel num_threads(8)
    {
        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + x0, 16);
        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + x0, 16);
        tmp2.store(out_ptr0 + x0, 16);
        auto tmp3 = in_ptr0[x0];
        out_ptr0[x0] = tmp3;
    }
}
\'\'\')

cpp_fused_sum_1 = async_compile.cpp_pybinding(['float*', 'const float*'], r\'\'\'
extern "C"  void  kernel(float* in_out_ptr0,
                       const float* in_ptr0)
{
    float tmp_acc0 = 0;
    auto tmp0 = in_ptr0[x1 + 4*x0];
    in_out_ptr0[x0] = tmp_acc0;
}
\'\'\')

async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    buf0 = empty_strided_cpu((512, 1024), (1024, 1), torch.float32)
    cpp_fused_add_mul_0(arg0_1, arg1_1, buf0)
    buf1 = empty_strided_cpu((512, ), (1, ), torch.float32)
    buf2 = buf1
    extern_kernels.mm(buf0, arg1_1, out=buf1)
    buf3 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(buf0, buf0, buf0)
    cpp_fused_sum_1(buf2, buf0)
    buf4 = empty_strided_cpu((s0, 8), (8, 1), torch.bfloat16)
    return (buf2,)
'''

TRITON_SRC = '''
triton_poi_fused_add_0 = async_compile.triton('triton_poi_fused_add_0', \'\'\'
import triton
@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tl.store(out_ptr0 + (x0), tmp2, xmask)
\'\'\', device_str='cuda')

triton_red_fused_sum_1 = async_compile.triton('triton_red_fused_sum_1', \'\'\'
@triton.jit
def triton_red_fused_sum_1(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    tmp0 = tl.load(in_ptr0 + (r1 + 4*x0), rmask & xmask)
    tl.store(in_out_ptr0 + (x0), tmp3, xmask)
\'\'\', device_str='cuda')

async_compile.wait(globals())

def call(args):
    buf0 = empty_strided_cuda((512, 1024), (1024, 1), torch.float16)
    triton_poi_fused_add_0.run(arg0_1, arg1_1, buf0, 524288, stream=stream0)
    extern_kernels.addmm(arg2_1, buf0, arg1_1, alpha=1, beta=1, out=buf1)
    triton_red_fused_sum_1.run(buf1, buf0, 512, 1024, stream=stream0)
    return (buf1,)
'''


def test_cpp_kernels_counted_and_classified():
    s = analyze_source(CPP_SRC)
    assert s.kernel_count == 2
    assert s.kernel_names() == ["cpp_fused_add_mul_0", "cpp_fused_sum_1"]
    k0, k1 = s.kernels
    assert (k0.n_in_ptr, k0.n_out_ptr, k0.n_inout_ptr) == (2, 1, 0)
    assert (k1.n_in_ptr, k1.n_out_ptr, k1.n_inout_ptr) == (1, 0, 1)
    assert k0.category == "pointwise" and k1.category == "reduction"
    assert k0.vectorized and not k1.vectorized
    assert k0.n_omp_parallel == 1
    assert k0.n_loads == 3 and k0.n_stores == 2
    assert k1.n_loads == 1 and k1.n_stores == 1
    assert s.pointwise_kernels == 1 and s.reduction_kernels == 1


def test_cpp_calls_externs_and_allocs():
    s = analyze_source(CPP_SRC)
    assert s.kernel_calls == 2 and all(k.n_calls == 1 for k in s.kernels)
    assert s.extern_calls == {"extern_kernels.mm": 1, "torch.ops.aten._scaled_dot_product_flash_attention_for_cpu": 1}
    assert s.extern_call_count == 2
    # three allocations; the symbolic one (s0, 8) contributes a count but no bytes
    assert s.alloc_count == 3
    assert s.alloc_bytes == 512 * 1024 * 4 + 512 * 4


def test_triton_kernels_parsed_the_same_way():
    s = analyze_source(TRITON_SRC)
    assert s.kernel_names() == ["triton_poi_fused_add_0", "triton_red_fused_sum_1"]
    assert [k.category for k in s.kernels] == ["pointwise", "reduction"]
    assert s.kernels[0].n_loads == 2 and s.kernels[0].n_stores == 1
    assert s.kernels[1].n_inout_ptr == 1
    assert s.kernel_calls == 2
    assert s.extern_calls == {"extern_kernels.addmm": 1}
    assert s.alloc_bytes == 512 * 1024 * 2


def test_aggregate_over_sources_and_roundtrip():
    s = analyze_sources([CPP_SRC, TRITON_SRC])
    assert s.n_sources == 2 and s.kernel_count == 4 and s.extern_call_count == 3
    from rewrite_blame.codegen_stats import CodeStats
    s2 = CodeStats.from_dict(s.to_dict())
    assert s2.kernel_names() == s.kernel_names() and s2.alloc_bytes == s.alloc_bytes


def test_diff_ignores_numeric_suffix_and_reports_new_kernels():
    a = analyze_source(CPP_SRC)
    b = analyze_source(CPP_SRC.replace("cpp_fused_sum_1", "cpp_fused_sum_7").replace("cpp_fused_add_mul_0", "cpp_fused_add_0"))
    d = diff_stats(a, b)
    assert d["kernel_count"] == (2, 2)
    assert d["kernels_only_in_a"] == ["cpp_fused_add_mul"] and d["kernels_only_in_b"] == ["cpp_fused_add"]


def test_empty_source_gives_zero_stats():
    s = analyze_source("def call(args):\n    return ()\n")
    assert s.kernel_count == 0 and s.alloc_count == 0 and s.extern_call_count == 0
