#pragma once
#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Result from a single VLM inference run.
@interface VLMInferenceResult : NSObject
@property (nonatomic, copy)   NSString *text;
@property (nonatomic, assign) double ttftMs;           ///< time-to-first-token, milliseconds
@property (nonatomic, assign) double decodeTokensPerSec; ///< steady-state decode throughput
@property (nonatomic, assign) NSUInteger totalTokens;
@property (nonatomic, assign) double peakMemoryMB;     ///< physical memory footprint, MB
@end

/// Thin ObjC++ wrapper around llama.cpp + libmtmd for VLM inference on iOS.
/// Create once per model pair; thread-safe for reads, not for concurrent infer calls.
@interface LlamaVLMRunner : NSObject

/// modelPath  – path to the LLM .gguf file
/// mmprojPath – path to the mmproj .gguf file
- (nullable instancetype)initWithModelPath:(NSString *)modelPath
                               mmprojPath:(NSString *)mmprojPath
                                    error:(NSError *__autoreleasing *)error;

/// Run one inference pass.  imagePath may be nil for text-only.
- (nullable VLMInferenceResult *)inferWithImagePath:(nullable NSString *)imagePath
                                             prompt:(NSString *)prompt
                                          maxTokens:(NSInteger)maxTokens
                                              error:(NSError *__autoreleasing *)error;

@end

NS_ASSUME_NONNULL_END
