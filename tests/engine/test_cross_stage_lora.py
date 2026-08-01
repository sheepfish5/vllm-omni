# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for cross-stage LoRA routing in the orchestrator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestBuildEngineCoreRequestLoRA:
    """Verify build_engine_core_request_from_tokens passes LoRA from params."""

    def test_lora_extracted_from_diffusion_params(self):
        lr = LoRARequest(lora_name="test", lora_int_id=1, lora_path="/tmp/fake")
        params = OmniDiffusionSamplingParams(lora_request=lr)

        # OmniDiffusionSamplingParams is not a SamplingParams, so
        # build_engine_core_request_from_tokens takes the pooling path.
        # We only care that lora_request is extracted via getattr.
        request = build_engine_core_request_from_tokens(
            request_id="req-1",
            prompt={"prompt_token_ids": [1, 2, 3]},
            params=params,
            model_config=None,
        )
        assert request.lora_request is lr

    def test_no_lora_on_sampling_params(self):
        params = SamplingParams(max_tokens=10)

        request = build_engine_core_request_from_tokens(
            request_id="req-2",
            prompt={"prompt_token_ids": [1, 2, 3]},
            params=params,
            model_config=None,
        )
        assert request.lora_request is None


class TestBuildEngineCoreRequestSamplingParams:
    def test_resolves_eos_like_vllm_input_processor(self):
        params = SamplingParams(max_tokens=10)
        model_config = SimpleNamespace(
            max_model_len=128,
            try_get_generation_config=lambda: {},
        )
        tokenizer = SimpleNamespace(eos_token_id=151645)

        request = build_engine_core_request_from_tokens(
            request_id="req-eos",
            prompt={"prompt_token_ids": [1, 2, 3]},
            params=params,
            model_config=model_config,
            tokenizer=tokenizer,
        )

        assert request.sampling_params is not None
        assert request.sampling_params.eos_token_id == 151645
        # The request-local clone is updated; deploy defaults remain reusable.
        assert params.eos_token_id is None

    def test_ignore_eos_remains_effective(self):
        params = SamplingParams(max_tokens=10, ignore_eos=True)
        model_config = SimpleNamespace(
            max_model_len=128,
            try_get_generation_config=lambda: {},
        )

        request = build_engine_core_request_from_tokens(
            request_id="req-ignore-eos",
            prompt={"prompt_token_ids": [1, 2, 3]},
            params=params,
            model_config=model_config,
            tokenizer=SimpleNamespace(eos_token_id=151645),
        )

        assert request.sampling_params is not None
        assert request.sampling_params.eos_token_id is None
