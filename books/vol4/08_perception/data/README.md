# Camera bandwidth illustration

`camera_bandwidth_scaling.csv` contains constructed camera formats, not historical deployments or measured bus traffic. The chapter cell computes ideal packed raw ingress as `width × height × bits_per_pixel × frame_rate / 8 / 10^6` MB/s. It excludes blanking, link protocol overhead, ISP reads/writes, copies, burst shape, and arbitration delay. The 1080p and 4K rows are the format assumptions used in the camera-ingress example.
