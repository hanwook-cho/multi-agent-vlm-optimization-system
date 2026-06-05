#import "LlamaVLMRunner.h"

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <mach/mach.h>
#include <chrono>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static uint64_t physicalFootprintBytes() {
    task_vm_info_data_t vmInfo;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO,
                                 (task_info_t)&vmInfo, &count);
    if (kr == KERN_SUCCESS) {
        return vmInfo.phys_footprint;
    }
    // fallback to resident_size
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
        return info.resident_size;
    }
    return 0;
}

// Suppress noisy llama.cpp logs in production builds
static void silentLogCallback(ggml_log_level level, const char * text, void * user_data) {
#if DEBUG
    if (level >= GGML_LOG_LEVEL_WARN) {
        NSLog(@"[llama.cpp] %s", text);
    }
#else
    (void)level; (void)text; (void)user_data;
#endif
}

// ---------------------------------------------------------------------------
// VLMInferenceResult
// ---------------------------------------------------------------------------

@implementation VLMInferenceResult
@end

// ---------------------------------------------------------------------------
// LlamaVLMRunner
// ---------------------------------------------------------------------------

@implementation LlamaVLMRunner {
    llama_model   * _model;
    llama_context * _ctx;
    mtmd_context  * _mtmd;
}

- (void)dealloc {
    if (_mtmd)  { mtmd_free(_mtmd);            _mtmd  = nullptr; }
    if (_ctx)   { llama_free(_ctx);            _ctx   = nullptr; }
    if (_model) { llama_model_free(_model);    _model = nullptr; }
}

- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                               mmprojPath:(NSString *)mmprojPath
                                    error:(NSError *__autoreleasing *)error {
    self = [super init];
    if (!self) return nil;

    // Silence noisy logs
    llama_log_set(silentLogCallback, nullptr);
    mtmd_log_set(silentLogCallback, nullptr);

    // 1. Load the LLM
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99; // offload all layers to Metal
    _model = llama_model_load_from_file(modelPath.UTF8String, mparams);
    if (!_model) {
        if (error) *error = [NSError errorWithDomain:@"LlamaVLMRunner"
                                                code:1
                                            userInfo:@{NSLocalizedDescriptionKey:
                                                           @"Failed to load LLM model"}];
        return nil;
    }

    // 2. Create llama context
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = 4096;
    cparams.n_batch  = 512;
    cparams.n_ubatch = 512;
    _ctx = llama_new_context_with_model(_model, cparams);
    if (!_ctx) {
        if (error) *error = [NSError errorWithDomain:@"LlamaVLMRunner"
                                                code:2
                                            userInfo:@{NSLocalizedDescriptionKey:
                                                           @"Failed to create llama context"}];
        return nil;
    }

    // 3. Load mmproj / mtmd context
    mtmd_context_params mctx = mtmd_context_params_default();
    mctx.use_gpu       = true;
    mctx.print_timings = false;
    _mtmd = mtmd_init_from_file(mmprojPath.UTF8String, _model, mctx);
    if (!_mtmd) {
        if (error) *error = [NSError errorWithDomain:@"LlamaVLMRunner"
                                                code:3
                                            userInfo:@{NSLocalizedDescriptionKey:
                                                           @"Failed to load mmproj / init mtmd context"}];
        return nil;
    }

    return self;
}

- (nullable VLMInferenceResult *)inferWithImagePath:(nullable NSString *)imagePath
                                             prompt:(NSString *)prompt
                                          maxTokens:(NSInteger)maxTokens
                                              error:(NSError *__autoreleasing *)error {

    // --- memory baseline ---
    uint64_t memBefore = physicalFootprintBytes();

    // --- build the input text with the image marker ---
    std::string marker = mtmd_default_marker();
    std::string fullPrompt;
    bool hasImage = (imagePath != nil && imagePath.length > 0);

    // Build chat-formatted prompt matching the model's template
    // LFM2-VL uses ChatML format: <|im_start|>user\n<img>\nQuestion<|im_end|>\n<|im_start|>assistant\n
    if (hasImage) {
        fullPrompt = std::string("<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
                                 "<|im_start|>user\n") + marker + "\n"
                   + std::string(prompt.UTF8String) + "<|im_end|>\n<|im_start|>assistant\n";
    } else {
        fullPrompt = std::string("<|im_start|>system\nYou are a helpful assistant<|im_end|>\n"
                                 "<|im_start|>user\n")
                   + std::string(prompt.UTF8String) + "<|im_end|>\n<|im_start|>assistant\n";
    }

    // --- load image as bitmap ---
    mtmd_bitmap * bitmap = nullptr;
    if (hasImage) {
        bitmap = mtmd_helper_bitmap_init_from_file(_mtmd, imagePath.UTF8String);
        if (!bitmap) {
            if (error) *error = [NSError errorWithDomain:@"LlamaVLMRunner"
                                                    code:4
                                                userInfo:@{NSLocalizedDescriptionKey:
                                                               @"Failed to load image bitmap"}];
            return nil;
        }
    }

    // --- tokenize ---
    mtmd_input_text inputText = {};
    inputText.text          = fullPrompt.c_str();
    inputText.add_special   = false;
    inputText.parse_special = true;

    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bitmaps[] = { bitmap };
    int32_t tokErr = mtmd_tokenize(_mtmd, chunks,
                                   &inputText,
                                   hasImage ? bitmaps : nullptr,
                                   hasImage ? 1 : 0);
    if (bitmap) { mtmd_bitmap_free(bitmap); bitmap = nullptr; }

    if (tokErr != 0) {
        mtmd_input_chunks_free(chunks);
        if (error) *error = [NSError errorWithDomain:@"LlamaVLMRunner"
                                                code:5
                                            userInfo:@{NSLocalizedDescriptionKey:
                                                           @"mtmd_tokenize failed"}];
        return nil;
    }

    // --- reset KV cache ---
    llama_memory_clear(llama_get_memory(_ctx), false);
    llama_pos n_past = 0;

    // --- encode + prefill all chunks ---
    int32_t evalErr = mtmd_helper_eval_chunks(_mtmd, _ctx, chunks, n_past,
                                              /*seq_id*/0, /*n_batch*/512,
                                              /*logits_last*/true, &n_past);
    mtmd_input_chunks_free(chunks);

    if (evalErr != 0) {
        if (error) *error = [NSError errorWithDomain:@"LlamaVLMRunner"
                                                code:6
                                            userInfo:@{NSLocalizedDescriptionKey:
                                                           @"Prefill eval failed"}];
        return nil;
    }

    // --- decode loop — measure TTFT and decode speed ---
    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.1f));
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    std::string outputText;
    uint64_t    peakMem = physicalFootprintBytes();
    double      ttftMs  = -1.0;
    NSInteger   nTokens = 0;

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_first = t_start; // will be set on first token

    llama_token eosToken = llama_vocab_eos(llama_model_get_vocab(_model));

    for (NSInteger i = 0; i < maxTokens; i++) {
        llama_token tok = llama_sampler_sample(sampler, _ctx, -1);
        llama_sampler_accept(sampler, tok);

        if (tok == eosToken || llama_vocab_is_eog(llama_model_get_vocab(_model), tok)) {
            break;
        }

        // Record TTFT on the very first token
        if (i == 0) {
            t_first = std::chrono::high_resolution_clock::now();
            ttftMs  = std::chrono::duration<double, std::milli>(t_first - t_start).count();
        }

        // Decode the token to text
        char piece[256] = {};
        llama_token_to_piece(llama_model_get_vocab(_model), tok, piece, sizeof(piece) - 1, 0, true);
        outputText += piece;
        nTokens++;

        // Track peak memory
        uint64_t mem = physicalFootprintBytes();
        if (mem > peakMem) peakMem = mem;

        // Prepare next batch
        llama_batch batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(_ctx, batch) != 0) { break; }
        n_past++;
    }

    llama_sampler_free(sampler);

    auto t_end = std::chrono::high_resolution_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    // Decode TPS = tokens generated / time from first token to end (excludes TTFT/prefill)
    double decodeSec = (totalMs - ttftMs) / 1000.0;
    double tps = (nTokens > 1 && decodeSec > 0) ? (double)(nTokens - 1) / decodeSec : 0.0;

    double peakMemMB = (double)(MAX(peakMem, memBefore) - memBefore) / (1024.0 * 1024.0);
    // Always report at least the model's working set
    if (peakMemMB < 0) peakMemMB = (double)peakMem / (1024.0 * 1024.0);

    VLMInferenceResult * result  = [[VLMInferenceResult alloc] init];
    result.text                  = [NSString stringWithUTF8String:outputText.c_str()];
    result.ttftMs                = ttftMs;
    result.decodeTokensPerSec    = tps;
    result.totalTokens           = (NSUInteger)nTokens;
    result.peakMemoryMB          = (double)peakMem / (1024.0 * 1024.0);
    return result;
}

@end
