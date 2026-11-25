#!/bin/bash
#SBATCH --job-name=space_audit3
#SBATCH --account=public-blaufer_v2
#SBATCH --partition=power-general-shared-pool
#SBATCH --constraint=ib_dj
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/scratch200/yotams2/logs/%x_%j.out
# 11-25: sizes of scratch dirs + moshe backups of the mono corpora (read-only; deletes nothing).
M=/moshe-data/blaufer-data/yotams2; S=/scratch200/yotams2
echo "== quota"; quota -s -f /scratch200
echo "== scratch top"; du -sh $S/* 2>/dev/null
echo "== sac-ssamba/data"; du -sh $S/sac-ssamba/data/* 2>/dev/null | sort -h | tail -15
echo "== sac-ssamba/data/mono"; ls -la $S/sac-ssamba/data/mono; du -sh $S/sac-ssamba/data/mono/* $S/sac-ssamba/data/mono/*/* 2>/dev/null
echo "== LibriSpeech IEMOCAP VoxCeleb1 SpeechCommands subdirs"; du -sh $S/sac-ssamba/data/{LibriSpeech,IEMOCAP,VoxCeleb1,SpeechCommands}/* 2>/dev/null
echo "== sac-ssamba/exp"; du -sh $S/sac-ssamba/exp/* 2>/dev/null | sort -h | tail -12
echo "== SAC-SAR-SSL/exp"; du -sh $S/SAC-SAR-SSL/exp/* 2>/dev/null | sort -h
echo "== joint_pretrain"; du -sh $S/SAC-SAR-SSL/exp/joint_pretrain/* 2>/dev/null | sort -h
echo "== joint_pretrain weights"; ls -la $S/SAC-SAR-SSL/exp/joint_pretrain/*/*.pth 2>/dev/null | awk '{print $5, $9}'
echo "== moshe data"; ls -la $M $M/data 2>&1; ls -la $M/data/* 2>&1 | head -60
echo "== moshe sac-ssamba"; ls -la $M/sac-ssamba 2>&1 | head -30
echo "$(date) done"
