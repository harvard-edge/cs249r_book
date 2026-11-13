# Chapter 8 Data Provenance

Chapter 8 (`08_perception.qmd`) stores no local Tier-2 `.csv` tables in this directory. All sensor datasheet specifications (navigation camera, wrist camera, Ouster OS1-64 Rev 7 LiDAR, IMU) and scenario timing/error budgets (`WarehouseAisle`, `WarehouseNavCameraAge`, `WarehouseNavCameraRollingShutter`, `WarehouseLidarSweep`, `WarehouseLongAisleMountError`, `WarehouseLidarExtrinsicShift`, `WarehouseStereoRigDepthError`, `AluminumLinkWarming`) are loaded from the `mlsysim` registry (`Embodied.MobileManipulator.WarehouseMobileManipulator`, `Embodied.Scenario.WarehouseAisle`, and `Scenarios.*`) and guarded in the chapter's LEGO calculation cells.

An earlier synthetic format table (`camera_bandwidth_scaling.csv`) was retired; camera ingress rates are computed analytically from configured resolution, bit depth, and frame rate in `calc-perception-camera-bandwidth`.
