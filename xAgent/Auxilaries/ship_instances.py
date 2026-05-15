from xAgent.Environment.agent import ShipParams

#Values should merely be plausible, model is heavily approximated anyways.
def dinghy() -> ShipParams:
  return ShipParams(
    max_curvature=1 / 8,
    max_velocity=4.0,
    max_reverse_velocity=1.5,
    length=3.5,
    width=1.4,
    tau=1.5
  )


def fishing_boat() -> ShipParams:
  return ShipParams(
    max_curvature=1 / 40,
    max_velocity=6.0,
    max_reverse_velocity=2.0,
    length=12.0,
    width=3.5,
    tau=4.0
  )


def tugboat() -> ShipParams:
  return ShipParams(
    max_curvature=1 / 70,
    max_velocity=5.0,
    max_reverse_velocity=3.0,
    length=30.0,
    width=10.0,
    tau=8.0
  )


def patrol_boat() -> ShipParams:
  return ShipParams(
    max_curvature=1 / 100,
    max_velocity=15.0,
    max_reverse_velocity=4.0,
    length=25.0,
    width=5.0,
    tau=3.0
  )


def ferry() -> ShipParams:
  return ShipParams(
    max_curvature=1 / 180,
    max_velocity=10.0,
    max_reverse_velocity=2.5,
    length=80.0,
    width=18.0,
    tau=12.0
  )


def cargo_ship() -> ShipParams:
  return ShipParams(
    max_curvature=1 / 900,
    max_velocity=11.0,
    max_reverse_velocity=1.0,
    length=220.0,
    width=32.0,
    tau=35.0
  )


def container_ship() -> ShipParams:
  return ShipParams(
    max_curvature=1 / 1400,
    max_velocity=13.0,
    max_reverse_velocity=0.8,
    length=350.0,
    width=45.0,
    tau=50.0
  )


def oil_tanker() -> ShipParams:
  return ShipParams(
    max_curvature=1 / 2500,
    max_velocity=9.0,
    max_reverse_velocity=0.5,
    length=400.0,
    width=60.0,
    tau=80.0
  )