"""The Rounded cube."""

from dm_control.entities.props import primitive
from dm_control import mjcf

from dexterity import _SRC_ROOT
from pathlib import Path

_TEXTURE_PATH:Path = _SRC_ROOT / "manipulation" / \
  "props" / "rounded_cube" / "rounded_cube.png"
_MESH_PATH:Path = _SRC_ROOT / "manipulation" / \
  "props" / "rounded_cube" / "rounded_cube.stl"

class MyPrimitive(primitive.Primitive):
  def _build(self, geom_type, size, name=None, **kwargs):
    self._mjcf_root = mjcf.element.RootElement(model=name)
    if 'mesh' in kwargs:
      kwargs['mesh'] = self._mjcf_root.asset.add(
        "mesh",
        name="mesh",
        file=str(kwargs['mesh']),)
    self._geom = self._mjcf_root.worldbody.add(
        'geom', name='geom', type=geom_type, size=size, **kwargs)
    self._position = self._mjcf_root.sensor.add(
        'framepos', name='position', objtype='geom', objname=self.geom)
    self._orientation = self._mjcf_root.sensor.add(
        'framequat', name='orientation', objtype='geom', objname=self.geom)
    self._linear_velocity = self._mjcf_root.sensor.add(
        'framelinvel', name='linear_velocity', objtype='geom',
        objname=self.geom)
    self._angular_velocity = self._mjcf_root.sensor.add(
        'frameangvel', name='angular_velocity', objtype='geom',
        objname=self.geom)

class RoundedCube(MyPrimitive):
  """A cube with Rounded letters on each face."""

  def _build(
      self,
      size: float,
      name: str = "rounded_cube",
  ) -> None:
    """Builds the cube.

    Args:
        size: The half-length of the cube.
        name: Optional name for the cube prop.
    """
    super()._build(geom_type="mesh", size=[size] * 3, name=name, mesh=str(_MESH_PATH), friction=[100,0.01,0.05], density=5000)

    self.mjcf_model.asset.add(
      "texture",
      name="texture_rounded",
      file=str(_TEXTURE_PATH),
      gridsize="3 4",
      gridlayout=".U..LFRB.D..",
    )
    self.mjcf_model.asset.add(
      "material",
      name="material_rounded",
      texture="texture_rounded",
      specular="1",
      shininess=".3",
      reflectance="0.0",
      rgba="1 1 1 1",
    )

    setattr(self.mjcf_model.find("geom", "geom"), "material", "material_rounded")

  @property
  def name(self) -> str:
    return self.mjcf_model.model

