import os
from pathlib import Path

import dotenv
import pytest

from ai_diffusion import workflow
from ai_diffusion.api import TextInput
from ai_diffusion.api import WorkflowKind, WorkflowInput, ControlInput
from ai_diffusion.client import Client, ClientEvent, CheckpointInfo
from ai_diffusion.comfy_client import ComfyClient
from ai_diffusion.image import Extent, Image
from ai_diffusion.resources import ControlMode
from ai_diffusion.settings import PerformanceSettings
from ai_diffusion.style import SDVersion, Style
from .config import result_dir, default_checkpoint, validation_dir, root_dir

client_params = ["local"]


@pytest.fixture(params=client_params, scope="module")
def client(request, qtapp):
    dotenv.load_dotenv(root_dir / "service" / ".env.local")
    url = os.environ["TEST_COMFYUI_URL"]
    return qtapp.run(ComfyClient.connect(url))


default_seed = 1234
default_perf = PerformanceSettings(batch_size=8)


def default_style(client: Client, sd_ver=SDVersion.sd15):
    version_checkpoints = [c for c in client.models.checkpoints if sd_ver.matches(c)]
    checkpoint = default_checkpoint[sd_ver]

    style = Style(Path("default.json"))
    style.sd_checkpoint = (
        checkpoint if checkpoint in version_checkpoints else version_checkpoints[0]
    )
    return style


def create(kind: WorkflowKind, client: Client, **kwargs):
    kwargs.setdefault("text", TextInput(""))
    kwargs.setdefault("style", default_style(client))
    kwargs.setdefault("seed", default_seed)
    kwargs.setdefault("perf", default_perf)
    return workflow.prepare(kind, models=client.models, **kwargs)


async def receive_images(client: Client, work: WorkflowInput):
    job_id = None
    async for msg in client.listen():
        if not job_id:
            job_id = await client.enqueue(work)
        if msg.event is ClientEvent.finished and msg.job_id == job_id:
            assert msg.images is not None
            return msg.images
        if msg.event is ClientEvent.error and msg.job_id == job_id:
            raise Exception(msg.error)
    assert False, "Connection closed without receiving images"


def run_and_save(
        qtapp,
        client: Client,
        work: WorkflowInput,
        filename: str,
        output_dir: Path = result_dir,
):
    dump_workflow(work, filename, client)

    async def runner():
        return await receive_images(client, work)

    results = qtapp.run(runner())
    assert len(results) == default_perf.batch_size
    client_name = "local"
    for idx, image in enumerate(results):
        image.save(output_dir / f"{filename}_{client_name}_{idx}.png")

    return results[0]


def dump_workflow(work: WorkflowInput, filename: str, client: Client):
    flow = workflow.create(work, client.models)
    flow.dump((result_dir / "workflows" / filename).with_suffix(".json"))


def test_control_pose_generate(qtapp, client):
    client.models.checkpoints = {
        "v3/RD_D_rdgirl_rdgirl_v3_1_3.safetensors": CheckpointInfo("RD_D_rdgirl_rdgirl_v3_1_3", SDVersion.sd15)}
    pose_image = Image.load(validation_dir / "[Control] Pose.png")
    extent = Extent(500, 500)
    prompt = TextInput("iw202, running")
    control = [ControlInput(ControlMode.pose, pose_image, 1.0)]
    job = create(
        WorkflowKind.generate, client, perf=default_perf, canvas=extent, text=prompt, control=control
    )
    result = run_and_save(qtapp, client, job, f"test_control_pose_generate_{extent.width}x{extent.height}")
    assert result.extent == extent
