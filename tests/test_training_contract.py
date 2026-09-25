from argparse import Namespace

import pytest
import torch

from scripts.train import (
    assert_finite_post_update_state,
    build_training_contract,
    parse_args,
    validate_full_batch_training_set,
    validate_training_args,
)


def make_model_and_optimizer() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_cli_rejects_zero_negative_and_nonfinite_training_options() -> None:
    invalid_arguments = (
        ["--steps", "0"],
        ["--steps", "-1"],
        ["--batch-size", "0"],
        ["--threads", "0"],
        ["--learning-rate", "0"],
        ["--learning-rate", "-0.1"],
        ["--learning-rate", "nan"],
        ["--learning-rate", "inf"],
    )
    for arguments in invalid_arguments:
        with pytest.raises(SystemExit):
            parse_args(arguments)


def test_runtime_validation_covers_programmatic_arguments_and_empty_loader() -> None:
    with pytest.raises(ValueError, match="steps must be an integer >= 1"):
        validate_training_args(
            Namespace(steps=0, batch_size=1, threads=1, learning_rate=1e-3)
        )
    with pytest.raises(ValueError, match="learning-rate must be a finite"):
        validate_training_args(
            Namespace(
                steps=1,
                batch_size=1,
                threads=1,
                learning_rate=float("inf"),
            )
        )
    with pytest.raises(ValueError, match="drop_last=True would yield no"):
        validate_full_batch_training_set(num_examples=3, batch_size=4)


def test_post_update_check_rejects_model_and_optimizer_corruption() -> None:
    model, optimizer = make_model_and_optimizer()
    loss = model(torch.ones(1, 2)).square().sum()
    loss.backward()
    optimizer.step()
    assert_finite_post_update_state(model, optimizer, step=1)

    first_parameter = next(model.parameters())
    with torch.no_grad():
        first_parameter.view(-1)[0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite post-update.*model"):
        assert_finite_post_update_state(model, optimizer, step=2)

    with torch.no_grad():
        first_parameter.view(-1)[0] = 0.0
        first_state = next(iter(optimizer.state.values()))
        first_state["exp_avg"].view(-1)[0] = float("inf")
    with pytest.raises(FloatingPointError, match="non-finite post-update.*optimizer"):
        assert_finite_post_update_state(model, optimizer, step=2)


def test_success_flags_are_derived_from_completed_checked_updates() -> None:
    model, optimizer = make_model_and_optimizer()
    no_step = build_training_contract(
        requested_steps=1,
        successful_steps=0,
        post_update_checks=0,
        examples_seen=0,
        losses=[],
        gradient_norms=[],
        model=model,
        optimizer=optimizer,
    )
    assert not no_step["real_audio_optimizer_step"]
    assert not no_step["nan_free"]
    assert not no_step["all_requested_steps_completed"]

    loss = model(torch.ones(1, 2)).square().sum()
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    assert_finite_post_update_state(model, optimizer, step=1)
    completed = build_training_contract(
        requested_steps=1,
        successful_steps=1,
        post_update_checks=1,
        examples_seen=1,
        losses=[float(loss.detach())],
        gradient_norms=[float(gradient_norm)],
        model=model,
        optimizer=optimizer,
    )
    assert completed["real_audio_optimizer_step"]
    assert completed["nan_free"]
    assert completed["all_requested_steps_completed"]
    assert completed["post_update_model_state_finite"]
    assert completed["post_update_optimizer_state_finite"]
