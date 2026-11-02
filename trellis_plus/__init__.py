"""Local3D+ : mesh, topology, LOD and texture quality mods for ComfyUI's native Pixal3D / TRELLIS.2 nodes."""
from typing_extensions import override

from comfy_api.latest import ComfyExtension, IO

from .mesh_ops import (TrellisPlusMeshRepair, TrellisPlusIsotropicRemesh, TrellisPlusQuadRemesh,
                       TrellisPlusSaveQuadOBJ)
from .texture_ops import TrellisPlusProjectImageTexture, TrellisPlusTextureEnhance


class TrellisPlusExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[IO.ComfyNode]]:
        return [TrellisPlusMeshRepair, TrellisPlusIsotropicRemesh, TrellisPlusQuadRemesh, TrellisPlusSaveQuadOBJ,
                TrellisPlusProjectImageTexture, TrellisPlusTextureEnhance]


async def comfy_entrypoint() -> TrellisPlusExtension:
    return TrellisPlusExtension()
