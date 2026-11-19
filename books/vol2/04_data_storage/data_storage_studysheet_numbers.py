"""Numbers for the Data Storage study sheet (Volume II, Chapter 4).

Every value the sheet prints is computed here from mlsysim, as in the chapter's LEGO cells,
and written as \\setnum{key}{value} to data_storage_studysheet_numbers.tex next to this file.
Inputs the sheet chooses for its own examples and questions are marked ASSUMED.
A printed result is computed from the rounded values the sheet prints beside it, and checked
against the unrounded model value. All storage units are decimal (GB = 1e9 bytes).
"""
import math
from pathlib import Path

from mlsysim import Hardware, Models, Systems
from mlsysim.engine.scenarios import Scenarios as SC


def c(x):
    return f"{x:,.0f}"


def fx(x, nd):
    return f"{x:.{nd}f}"


def numbers() -> dict:
    n = {}
    GBs = "GB/s"

    # ---- Front picture: per-client bandwidth by tier vs per-node demand (chapter: StoragePyramidContext,
    #      TextImageBandwidth). Scopes: HBM per accelerator; DRAM, NVMe stripe, PFS share per node.
    hbm = Hardware.Cloud.H100.memory.bandwidth.to(GBs).magnitude
    dram = Hardware.Tech.Storage.HostDram.bandwidth.to(GBs).magnitude
    drives = Systems.Storage.LocalNvmeGen4_7T68x4.devices_per_node
    drive_bw = Hardware.Tech.Storage.NvmeGen4.bandwidth.to(GBs).magnitude
    stripe = drives * drive_bw
    pfs = Systems.Storage.PfsOneTbPerSecond.bandwidth.to(GBs).magnitude
    cluster = Systems.Clusters.Production_2K_NDR
    nodes = cluster.count
    per_node = cluster.node.accelerators_per_node
    pfs_share = pfs / nodes
    text, image = SC.TextInputDemand2K, SC.ImageInputDemand2K
    text_node = text.result.bandwidth.to(GBs).magnitude / nodes
    image_node = image.result.bandwidth.to(GBs).magnitude / nodes
    n.update(hbmbandwidth=fx(hbm / 1000, 2), drambandwidth=c(dram), stripebandwidth=c(stripe),
             drivecount=c(drives), drivebandwidth=c(drive_bw), pfsaggregate=c(pfs / 1000), clusternodes=c(nodes),
             pfsshare=fx(pfs_share, 1), imagenode=fx(image_node, 2), textnode=fx(text_node * 1000, 2))
    # raw values for the TikZ ladder (GB/s)
    n.update(ladderhbm=fx(hbm, 1), ladderdram=fx(dram, 1), ladderstripe=fx(stripe, 1), ladderpfs=fx(pfs_share, 3),
             ladderimage=fx(image_node, 4), laddertext=fx(text_node, 6))
    # Mistake 1: HBM vs one NVMe drive (chapter: StorageFuelLineContext.bw_ratio), and the text job's need
    one_drive = Hardware.Cloud.H100.storage.bandwidth.to(GBs).magnitude
    n.update(tierratio=c(hbm / one_drive), textgpus=c(text.consumers),
             textjob=c(text.result.bandwidth.to("MB/s").magnitude))

    # ---- Worked example: 175B checkpoint staged on local NVMe (chapter: Gpt3TieredCheckpoint2K)
    ck = SC.Gpt3TieredCheckpoint2K
    params_b = Models.Language.GPT3.parameters.to("param").magnitude / 1e9
    bpp = ck.bytes_per_param.to("byte/param").magnitude
    ckpt_gb = params_b * bpp                       # 175 x 14 = 2,450 GB
    assert abs(ckpt_gb - ck.result.checkpoint_bytes.to("GB").magnitude) < 1e-6
    shard = round(ckpt_gb / nodes, 2)              # printed 9.57 GB
    assert abs(shard - ck.result.per_node_shard.to("GB").magnitude) < 0.005
    local = ck.local_bandwidth.to(GBs).magnitude
    write = shard / local
    assert abs(write - ck.result.local_write_time.to("s").magnitude) < 0.001
    durable = ck.durable_bandwidth.to(GBs).magnitude
    drain = ckpt_gb / durable
    interval = ck.save_interval.to("s").magnitude
    pause_pct = round(write, 2) / interval * 100
    n.update(params=c(params_b), bytesperparam=c(bpp), ckptgb=c(ckpt_gb), shardgb=fx(shard, 2),
             localbw=c(local), writes=fx(write, 2), durablebw=c(durable / 1000), drains=fx(drain, 2),
             intervalmin=c(interval / 60), intervals=c(interval), pausepct=fx(pause_pct, 2))
    # Your turn: same save, PFS shared by ten concurrent jobs (chapter prose after the storm notebook)
    jobs = 10                                       # chapter: "the ten concurrent jobs"
    share = durable / jobs
    direct = ckpt_gb / share
    n.update(jobs=c(jobs), sharebw=c(share), directs=fx(direct, 1), directpct=fx(direct / interval * 100, 1),
             directratio=c(direct / round(write, 2)))

    # ---- Q1 ASSUMED: 512 GPUs, 192 images of 120 KB, 150 ms compute, target 0.9; decode keeps up with 60 GB/s
    q_gpus, q_imgs, q_kb, q_ms, q_eta, q_decode = 512, 192, 120, 150, 0.9, 60
    q_batch_mb = q_imgs * q_kb / 1000
    q_need = q_gpus * q_eta * q_batch_mb / (q_ms / 1000) / 1000
    q_nodes = q_gpus // per_node
    q_fair = q_nodes * round(pfs_share, 1)
    n.update(qgpus=c(q_gpus), qimages=c(q_imgs), qkb=c(q_kb), qms=c(q_ms), qeta=fx(q_eta, 1), qdecode=c(q_decode),
             qbatch=fx(q_batch_mb, 2), qneed=fx(q_need, 1), qnodes=c(q_nodes), qfair=c(q_fair),
             pershare=fx(round(pfs_share, 1), 1))

    # ---- Q2 ASSUMED: 90 percent hits, misses 40x slower (equation as the chapter's HotTierCacheExample)
    h, miss = 0.90, 40
    mean = h + (1 - h) * miss
    need = (miss - 2) / (miss - 1)                  # hit rate for E[T] = 2 T_hot (half the all-hot rate)
    n.update(hitpct=c(h * 100), missx=c(miss), missshare=fx(1 - h, 1), hit=fx(h, 1), meanservice=fx(mean, 1),
             allhotpct=c(1 / mean * 100), halfhit=fx(need * 100, 1))
    # Chapter's own case for the front: 95 percent hits, 100x misses -> 5.95, one-sixth
    cache = SC.HotTierCacheExample
    n.update(chhit=c(cache.hit_rate * 100), chmiss=c(cache.miss_penalty), chmean=fx(cache.result.mean_service, 2),
             chfrac=c(1 / cache.result.delivered_fraction))

    # ---- Q3 ASSUMED: 150 ms compute, PFS p99 450 ms, 30 MB decoded batch, eight GPUs per node;
    #      then a stage that averages 160 ms per batch
    p_ms, p99, p_mb, slow = 150, 450, 30, 160
    depth = math.ceil(p99 / p_ms)
    stall = (slow - p_ms) / slow * 100
    n.update(pms=c(p_ms), pnine=c(p99), pmb=c(p_mb), pgpus=c(per_node), depth=c(depth),
             pbuffer=c(depth * p_mb * per_node), slowms=c(slow), slowgap=c(slow - p_ms), stallpct=fx(stall, 2))

    # ---- Q4: the chapter's metadata meltdown (ImageFileMetadataMeltdown)
    melt = SC.ImageFileMetadataMeltdown
    n.update(files=f"{melt.files / 1e6:.0f}", shards=c(melt.shards), filesper=c(melt.files_per_shard),
             ratedbw=c(melt.rated_bandwidth.to(GBs).magnitude))

    # ---- Mistake 2: stream vs stage a 50 TB vision corpus (VisionCorpusStreamVsStage)
    plan = SC.VisionCorpusStreamVsStage
    ds_tb = plan.dataset_bytes.to("TB").magnitude
    egress = plan.egress_rate.to("dollar/GB").magnitude
    per_run = ds_tb * 1000 * plan.epochs * egress
    assert abs(per_run - plan.result.stream_egress_per_run) < 1e-6
    store = plan.result.source_storage_cost
    n.update(corpustb=c(ds_tb), epochs=c(plan.epochs), egress=fx(egress, 2), perrun=c(per_run),
             storeyear=c(store), localrate=c(plan.local_rate.to("dollar/TB/month").magnitude),
             objectrate=fx(plan.object_rate.to("dollar/GB/month").magnitude, 2),
             breakeven=fx(plan.result.break_even_epochs, 1),
             stageegress=c(ds_tb * 1000 * egress))   # staging reads the corpus once per run
    return n


if __name__ == "__main__":
    out = Path(__file__).with_suffix(".tex")
    out.write_text("% generated by " + Path(__file__).name + "; do not edit\n"
                   + "".join(f"\\setnum{{{k}}}{{{v}}}\n" for k, v in numbers().items()))
    print(out)
