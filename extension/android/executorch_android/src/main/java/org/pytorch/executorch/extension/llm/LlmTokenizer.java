/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.llm;

import com.facebook.jni.HybridData;
import com.facebook.jni.annotations.DoNotStrip;
import java.io.File;
import org.pytorch.executorch.ExecuTorchRuntime;
import org.pytorch.executorch.annotations.Experimental;

/**
 * LlmTokenizer is a wrapper around the tokenizers used in LLM.
 * It provides an interface to encode text into tokens and decode tokens into text.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
public class LlmTokenizer {
  private final HybridData mHybridData;

  @DoNotStrip
  private static native HybridData initHybrid(String tokenizerPath);

  /**
   * Constructs an LlmTokenizer with the given tokenizer path.
   *
   * @param tokenizerPath Path to the tokenizer file (e.g., tokenizer.model)
   */
  public LlmTokenizer(String tokenizerPath) {
    ExecuTorchRuntime runtime = ExecuTorchRuntime.getRuntime();

    File tokenizerFile = new File(tokenizerPath);
    if (!tokenizerFile.canRead() || !tokenizerFile.isFile()) {
      throw new RuntimeException("Cannot load tokenizer path " + tokenizerPath);
    }

    mHybridData = initHybrid(tokenizerPath);
  }

  /**
   * Encodes the input text into a sequence of tokens.
   *
   * @param text The input text to encode.
   * @return An array of tokens.
   */
  public long[] encode(String text) {
      return encode(text, 0, 0);
  }

  /**
   * Encodes the input text into a sequence of tokens with optional BOS/EOS.
   *
   * @param text The input text to encode.
   * @param bos Number of BOS tokens to prepend.
   * @param eos Number of EOS tokens to append.
   * @return An array of tokens.
   */
  public native long[] encode(String text, int bos, int eos);

  /**
   * Decodes a single token/pair using the tokenizer state logic (if any).
   * Note: This exposes the low-level decode API.
   *
   * @param prevToken The previous token.
   * @param token The current token.
   * @return The decoded string piece.
   */
  public native String decode(long prevToken, long token);

  /**
   * Resets the native object.
   */
  public void resetNative() {
    mHybridData.resetNative();
  }
}
