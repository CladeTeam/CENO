# Copyright (c) 2025-2026, CENO Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, average_precision_score
from tqdm import tqdm
import json
import os
from torch.utils.data import DataLoader
import pandas as pd

class VEPEvaluator:
    """
    Variant Effect Prediction 评估器
    
    支持两种任务类型：
    1. Regression: 回归任务，计算序列分数与适应度的相关性
    2. Classification: 分类任务，计算变异效应的分类性能
    """
    
    def __init__(self, 
                 model_batch_size: int = 16,
                 dataloader_batch_size: int = 1024,
                 task_type : int = 1,
                 num_workers: int = 8,
                 save_detailed: bool = False,
                 group_by_distance: bool = False,
                 distance_group_col: str = "tss_distance_group",
                 resume: bool = False,
                 resume_progress_every: int = 1000):
        """
        Args:
            model_batch_size: 模型 score_sequences 的批处理大小
            dataloader_batch_size: DataLoader 的批处理大小
            num_workers: DataLoader 的工作进程数
            save_detailed: 是否保存详细的序列评分结果
            group_by_distance: 是否按距离分组统计分类指标
            distance_group_col: 分组列名（默认 tss_distance_group）
        """
        self.model_batch_size = model_batch_size
        self.dataloader_batch_size = dataloader_batch_size
        self.num_workers = num_workers
        self.save_detailed = save_detailed
        self.task_type = task_type
        self.group_by_distance = bool(group_by_distance)
        self.distance_group_col = distance_group_col
        self.resume = bool(resume)
        self.resume_progress_every = int(resume_progress_every)

    def _init_resume_state(self, output_dir: str, header_cols: List[str]) -> Tuple[int, str, str]:
        os.makedirs(output_dir, exist_ok=True)
        scores_path = os.path.join(output_dir, "detailed_scores.tsv")
        progress_path = os.path.join(output_dir, "progress.json")

        processed = 0
        file_count = 0
        if os.path.exists(progress_path):
            try:
                with open(progress_path, "r", encoding="utf-8") as f:
                    processed = int(json.load(f).get("processed", 0))
            except Exception:
                processed = 0
        if os.path.exists(scores_path):
            try:
                with open(scores_path, "r", encoding="utf-8") as f:
                    file_count = max(sum(1 for _ in f) - 1, 0)
            except Exception:
                file_count = 0
        processed = max(processed, file_count)

        if not os.path.exists(scores_path):
            with open(scores_path, "w", encoding="utf-8") as f:
                f.write("\t".join(header_cols) + "\n")

        return processed, scores_path, progress_path

    def _append_rows(self, scores_path: str, rows: List[Dict[str, Any]], header_cols: List[str]) -> None:
        if not rows:
            return
        with open(scores_path, "a", encoding="utf-8") as f:
            for row in rows:
                values = [str(row.get(col, "")) for col in header_cols]
                f.write("\t".join(values) + "\n")

    def _save_progress(self, progress_path: str, processed: int, total: Optional[int] = None) -> None:
        payload = {"processed": int(processed)}
        if total is not None:
            payload["total"] = int(total)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _load_scores_for_metrics(self, scores_path: str, task_type: int) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
        df = pd.read_csv(scores_path, sep="\t")
        group_labels = None
        if task_type == 1:
            predictions = df["seq_score"].astype(np.float32).to_numpy()
            targets = df["fitness"].astype(np.float32).to_numpy()
            return predictions, targets, None
        if "delta_score" in df.columns:
            predictions = df["delta_score"].astype(np.float32).to_numpy()
        else:
            predictions = df["score"].astype(np.float32).to_numpy()
        targets = df["label"].astype(np.int32).to_numpy()
        if self.distance_group_col in df.columns:
            group_labels = df[self.distance_group_col].astype(str).tolist()
        return predictions, targets, group_labels
    def evaluate_gpn_msa(self, result: pd.DataFrame, output_dir: Optional[str]):
        if self.task_type == 1:
            # 回归任务
            # 提取指标
            predictions = np.array(result["score"], dtype=np.float32)
            targets = np.array(result["label"], dtype=np.float32)
            metrics = self._compute_regression_metrics(predictions, targets)
            detailed_data = []
            # 构建结果
            results = {
                "task_type": "regression",
                "evaluation_method": "direct_scoring",
                "metrics": metrics,
                "model_info":"gpn_msa"
            }
            
            # 保存结果
            if output_dir is not None:
                self._save_results(results, detailed_data, output_dir)
            
            # 打印主要指标
            self._print_regression_metrics(metrics)
            
            return results
        elif self.task_type == 2:
            # 分类任务
            # 计算指标
            delta_scores = np.array(result["score"], dtype=np.float32)
            targets = np.array(result["label"], dtype=np.int32)
            metrics = self._compute_classification_metrics(delta_scores, targets)
            detailed_data = []
            # 构建结果
            results = {
                "task_type": "classification",
                "evaluation_method": "delta_scoring_optimized",
                "metrics": metrics,
                "model_info": "gpn_msa",
                # "dataset_info": self._get_dataset_info(dataset)
            }
            
            # 保存结果
            if output_dir is not None:
                self._save_results(results, detailed_data, output_dir)
            
            # 打印主要指标
            self._print_classification_metrics(metrics)
            
            return results
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

    
    def evaluate(self, 
                model, 
                dataset, 
                output_dir: Optional[str] = None,
                progress_bar: bool = True) -> Dict[str, Any]:
        """
        执行 VEP 评估 - 根据数据集任务类型自动选择评估方式
        
        Args:
            model: 实现了 score_sequences 接口的模型
            dataset: 实现了统一输出格式的数据集  
            output_dir: 结果输出目录
            progress_bar: 是否显示进度条
            
        Returns:
            评估结果字典
        """
        if self.resume and output_dir is None:
            raise ValueError("resume=True requires output_dir for progress tracking.")

        if self.task_type == 1:
            return self._evaluate_regression(model, dataset, output_dir, progress_bar)
        elif self.task_type == 2:
            return self._evaluate_classification(model, dataset, output_dir, progress_bar)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    
    def _evaluate_regression(self, 
                           model, 
                           dataset, 
                           output_dir: Optional[str],
                           progress_bar: bool) -> Dict[str, Any]:
        """
        回归任务评估：计算序列分数与适应度的相关性
        适用于 DMS 数据等连续适应度预测任务
        """
        # 创建 DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=self.dataloader_batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        
        all_seq_scores = []
        all_fitness = []
        detailed_data = []

        resume_processed = 0
        scores_path = progress_path = None
        header_cols = ["idx", "fitness", "seq_score"]
        if self.resume and output_dir is not None:
            resume_processed, scores_path, progress_path = self._init_resume_state(output_dir, header_cols)
            if resume_processed >= len(dataset):
                preds, targets, _ = self._load_scores_for_metrics(scores_path, task_type=1)
                metrics = self._compute_regression_metrics(preds, targets)
                results = {
                    "task_type": "regression",
                    "evaluation_method": "direct_scoring",
                    "metrics": metrics,
                    "model_info": self._get_model_info(model),
                }
                self._save_results(results, [], output_dir)
                self._print_regression_metrics(metrics)
                return results
        
        # 收集数据
        idx = 0
        processed = 0
        for batch_data in tqdm(dataloader, disable=(not progress_bar), ncols=120, desc="Evaluating regression"):
            wt_seqs, mt_seqs, fitnesses = batch_data

            batch_size = len(mt_seqs)
            if self.resume and resume_processed > idx:
                if resume_processed >= idx + batch_size:
                    idx += batch_size
                    continue
                skip = resume_processed - idx
                wt_seqs = wt_seqs[skip:]
                mt_seqs = mt_seqs[skip:]
                fitnesses = fitnesses[skip:]
                idx = resume_processed
            
            # 模型打分 - 对变异序列评分
            seq_scores = model.score_sequences(mt_seqs, batch_size=self.model_batch_size)
            
            # 收集用于相关性分析的数据
            if self.resume:
                rows = []
                for fit, seq_score in zip(fitnesses, seq_scores):
                    rows.append({
                        "idx": idx,
                        "fitness": float(fit),
                        "seq_score": float(seq_score),
                    })
                    idx += 1
                self._append_rows(scores_path, rows, header_cols)
                processed += len(rows)
                if processed % self.resume_progress_every == 0:
                    self._save_progress(progress_path, resume_processed + processed, total=len(dataset))
            else:
                all_seq_scores.extend(seq_scores)
                all_fitness.extend(fitnesses)
            
            # 如果需要保存详细结果（resume 模式下已写入文件）
            if self.save_detailed and not self.resume:
                for wt_seq, mt_seq, fitness, seq_score in zip(wt_seqs, mt_seqs, fitnesses, seq_scores):
                    detailed_data.append({
                        'idx': idx,
                        # 'wt_seq': wt_seq,
                        # 'mt_seq': mt_seq,
                        'fitness': float(fitness),
                        'seq_score': float(seq_score)
                    })
                    idx += 1
        
        # 计算指标
        if self.resume:
            self._save_progress(progress_path, resume_processed + processed, total=len(dataset))
            predictions, targets, _ = self._load_scores_for_metrics(scores_path, task_type=1)
        else:
            predictions = np.array(all_seq_scores, dtype=np.float32)
            targets = np.array(all_fitness, dtype=np.float32)
        metrics = self._compute_regression_metrics(predictions, targets)
        
        # 构建结果
        results = {
            "task_type": "regression",
            "evaluation_method": "direct_scoring",
            "metrics": metrics,
            "model_info": self._get_model_info(model),
            # "dataset_info": self._get_dataset_info(dataset)
        }
        
        if self.save_detailed:
            results["detailed_scores"] = detailed_data
        
        # 保存结果
        if output_dir is not None:
            self._save_results(results, detailed_data, output_dir)
        
        # 打印主要指标
        self._print_regression_metrics(metrics)
        
        return results
    

    def _evaluate_classification(self, 
                               model, 
                               dataset, 
                               output_dir: Optional[str],
                               progress_bar: bool) -> Dict[str, Any]:
        """
        分类任务评估：计算变异效应预测的分类性能
        适用于 TraitGym 等基因组变异功能预测任务
        """
        # 检查是否支持优化的评估方式
        if hasattr(dataset, 'get_unique_reference_sequences'):
            return self._evaluate_classification_optimized(model, dataset, output_dir, progress_bar)
        else:
            return self._evaluate_classification_standard(model, dataset, output_dir, progress_bar)
    
    def _evaluate_classification_optimized(self, 
                                         model, 
                                         dataset, 
                                         output_dir: Optional[str],
                                         progress_bar: bool) -> Dict[str, Any]:
        """
        优化的分类评估：利用去重的参考序列减少计算量
        """
        # 1. 获取去重的参考序列并评分
        ref_seqs = dataset.get_unique_reference_sequences()
        if progress_bar:
            print(f'Scoring {len(ref_seqs)} unique reference sequences...')
        ref_scores = model.score_sequences(ref_seqs, batch_size=self.model_batch_size)

        ref_index_map = None
        try:
            ref_index_map = {ref: i for i, ref in enumerate(ref_seqs)}
        except Exception:
            ref_index_map = None
        
        # 2. 收集所有变异序列和对应的参考序列索引
        dataloader = DataLoader(
            dataset, 
            batch_size=self.dataloader_batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        group_labels = self._get_group_labels(dataset)
        
        var_seqs = []
        ref_indexes = []
        all_labels = []
        detailed_data = []

        resume_processed = 0
        scores_path = progress_path = None
        header_cols = ["idx", "label", "ref_score", "var_score", "delta_score"]
        if group_labels is not None:
            header_cols.append(self.distance_group_col)
        if self.resume and output_dir is not None:
            resume_processed, scores_path, progress_path = self._init_resume_state(output_dir, header_cols)
            if resume_processed >= len(dataset):
                preds, targets, groups = self._load_scores_for_metrics(scores_path, task_type=2)
                metrics = self._compute_classification_metrics(preds, targets)
                group_metrics = None
                if groups is not None:
                    group_metrics = self._compute_grouped_classification_metrics(preds, targets, groups)
                results = {
                    "task_type": "classification",
                    "evaluation_method": "delta_scoring_optimized",
                    "metrics": metrics,
                    "model_info": self._get_model_info(model),
                }
                if group_metrics:
                    results["group_metrics"] = {
                        "group_column": self.distance_group_col,
                        "metrics_by_group": group_metrics,
                    }
                self._save_results(results, [], output_dir)
                self._print_classification_metrics(metrics)
                return results
        
        idx = 0
        processed = 0
        all_delta_scores = []
        all_labels = []

        for batch_data in tqdm(dataloader, disable=(not progress_bar), ncols=120, desc="Evaluating classification"):
            wt_seqs, mt_seqs, labels = batch_data

            batch_size = len(mt_seqs)
            if self.resume and resume_processed > idx:
                if resume_processed >= idx + batch_size:
                    idx += batch_size
                    continue
                skip = resume_processed - idx
                wt_seqs = wt_seqs[skip:]
                mt_seqs = mt_seqs[skip:]
                labels = labels[skip:]
                idx = resume_processed

            # 找到对应的参考序列索引
            ref_idx_batch = []
            for wt_seq in wt_seqs:
                if ref_index_map is not None:
                    try:
                        ref_idx_batch.append(ref_index_map[wt_seq])
                    except Exception:
                        raise ValueError(f"Reference sequence not found in unique sequences.")
                else:
                    try:
                        ref_idx_batch.append(ref_seqs.index(wt_seq))
                    except ValueError:
                        raise ValueError(f"Reference sequence not found in unique sequences.")

            var_scores = model.score_sequences(mt_seqs, batch_size=self.model_batch_size)
            ref_idx_arr = np.array(ref_idx_batch, dtype=np.int32)
            delta_scores = np.array(var_scores) - np.array(ref_scores)[ref_idx_arr]

            if self.resume:
                rows = []
                for i, (label, ref_idx) in enumerate(zip(labels, ref_idx_arr)):
                    row = {
                        "idx": idx,
                        "label": int(label),
                        "ref_score": float(ref_scores[ref_idx]),
                        "var_score": float(var_scores[i]),
                        "delta_score": float(delta_scores[i]),
                    }
                    if group_labels is not None:
                        row[self.distance_group_col] = group_labels[idx]
                    rows.append(row)
                    idx += 1
                self._append_rows(scores_path, rows, header_cols)
                processed += len(rows)
                if processed % self.resume_progress_every == 0:
                    self._save_progress(progress_path, resume_processed + processed, total=len(dataset))
            else:
                all_delta_scores.extend(delta_scores)
                all_labels.extend(labels)
                if self.save_detailed:
                    for wt_seq, mt_seq, label, ref_idx, var_score, delta_score in zip(
                        wt_seqs, mt_seqs, labels, ref_idx_arr, var_scores, delta_scores
                    ):
                        row = {
                            'idx': idx,
                            'label': int(label),
                            'ref_score': float(ref_scores[ref_idx]),
                            'var_score': float(var_score),
                            'delta_score': float(delta_score)
                        }
                        if group_labels is not None:
                            row[self.distance_group_col] = group_labels[idx]
                        detailed_data.append(row)
                        idx += 1

        # 计算指标
        if self.resume:
            self._save_progress(progress_path, resume_processed + processed, total=len(dataset))
            delta_scores, targets, group_labels_from_file = self._load_scores_for_metrics(scores_path, task_type=2)
            group_labels = group_labels_from_file
        else:
            delta_scores = np.array(all_delta_scores, dtype=np.float32)
            targets = np.array(all_labels, dtype=np.int32)

        metrics = self._compute_classification_metrics(delta_scores, targets)
        group_metrics = None
        if group_labels is not None:
            group_metrics = self._compute_grouped_classification_metrics(delta_scores, targets, group_labels)
        
        # 构建结果
        results = {
            "task_type": "classification",
            "evaluation_method": "delta_scoring_optimized",
            "metrics": metrics,
            "model_info": self._get_model_info(model),
            # "dataset_info": self._get_dataset_info(dataset)
        }
        if group_metrics:
            results["group_metrics"] = {
                "group_column": self.distance_group_col,
                "metrics_by_group": group_metrics,
            }
        
        if self.save_detailed:
            results["detailed_scores"] = detailed_data
        
        # 保存结果
        if output_dir is not None:
            self._save_results(results, detailed_data, output_dir)
        
        # 打印主要指标
        self._print_classification_metrics(metrics)
        
        return results
    
    def _evaluate_classification_standard(self, 
                                        model, 
                                        dataset, 
                                        output_dir: Optional[str],
                                        progress_bar: bool) -> Dict[str, Any]:
        """
        标准分类评估：逐对计算参考和变异序列
        """
        dataloader = DataLoader(
            dataset, 
            batch_size=self.dataloader_batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        group_labels = self._get_group_labels(dataset)
        
        all_delta_scores = []
        all_labels = []
        detailed_data = []

        resume_processed = 0
        scores_path = progress_path = None
        header_cols = ["idx", "label", "ref_score", "var_score", "delta_score"]
        if group_labels is not None:
            header_cols.append(self.distance_group_col)
        if self.resume and output_dir is not None:
            resume_processed, scores_path, progress_path = self._init_resume_state(output_dir, header_cols)
            if resume_processed >= len(dataset):
                preds, targets, groups = self._load_scores_for_metrics(scores_path, task_type=2)
                metrics = self._compute_classification_metrics(preds, targets)
                group_metrics = None
                if groups is not None:
                    group_metrics = self._compute_grouped_classification_metrics(preds, targets, groups)
                results = {
                    "task_type": "classification",
                    "evaluation_method": "delta_scoring_standard",
                    "metrics": metrics,
                    "model_info": self._get_model_info(model),
                }
                if group_metrics:
                    results["group_metrics"] = {
                        "group_column": self.distance_group_col,
                        "metrics_by_group": group_metrics,
                    }
                self._save_results(results, [], output_dir)
                self._print_classification_metrics(metrics)
                return results
        
        idx = 0
        processed = 0
        for batch_data in tqdm(dataloader, disable=(not progress_bar), ncols=120, desc="Evaluating classification"):
            wt_seqs, mt_seqs, labels = batch_data

            batch_size = len(mt_seqs)
            if self.resume and resume_processed > idx:
                if resume_processed >= idx + batch_size:
                    idx += batch_size
                    continue
                skip = resume_processed - idx
                wt_seqs = wt_seqs[skip:]
                mt_seqs = mt_seqs[skip:]
                labels = labels[skip:]
                idx = resume_processed
            
            # 分别评分参考序列和变异序列
            ref_scores = model.score_sequences(wt_seqs, batch_size=self.model_batch_size)
            var_scores = model.score_sequences(mt_seqs, batch_size=self.model_batch_size)
            
            # 计算 delta scores
            delta_scores = np.array(var_scores) - np.array(ref_scores)
            if self.resume:
                rows = []
                for i, label in enumerate(labels):
                    row = {
                        "idx": idx,
                        "label": int(label),
                        "ref_score": float(ref_scores[i]),
                        "var_score": float(var_scores[i]),
                        "delta_score": float(delta_scores[i]),
                    }
                    if group_labels is not None:
                        row[self.distance_group_col] = group_labels[idx]
                    rows.append(row)
                    idx += 1
                self._append_rows(scores_path, rows, header_cols)
                processed += len(rows)
                if processed % self.resume_progress_every == 0:
                    self._save_progress(progress_path, resume_processed + processed, total=len(dataset))
            else:
                all_delta_scores.extend(delta_scores)
                all_labels.extend(labels)
            
            # 收集详细数据（resume 模式下已写入文件）
            if self.save_detailed and not self.resume:
                for wt_seq, mt_seq, label, ref_score, var_score, delta_score in zip(
                    wt_seqs, mt_seqs, labels, ref_scores, var_scores, delta_scores):
                    row = {
                        'idx': idx,
                        # 'wt_seq': wt_seq,
                        # 'mt_seq': mt_seq, 
                        'label': int(label),
                        'ref_score': float(ref_score),
                        'var_score': float(var_score),
                        'delta_score': float(delta_score)
                    }
                    if group_labels is not None:
                        row[self.distance_group_col] = group_labels[idx]
                    detailed_data.append(row)
                    idx += 1
        
        # 计算指标
        if self.resume:
            self._save_progress(progress_path, resume_processed + processed, total=len(dataset))
            delta_scores, targets, group_labels_from_file = self._load_scores_for_metrics(scores_path, task_type=2)
            group_labels = group_labels_from_file
        else:
            delta_scores = np.array(all_delta_scores, dtype=np.float32)
            targets = np.array(all_labels, dtype=np.int32)
        metrics = self._compute_classification_metrics(delta_scores, targets)
        group_metrics = None
        if group_labels is not None:
            group_metrics = self._compute_grouped_classification_metrics(delta_scores, targets, group_labels)
        
        # 构建结果
        results = {
            "task_type": "classification",
            "evaluation_method": "delta_scoring_standard",
            "metrics": metrics,
            "model_info": self._get_model_info(model),
            # "dataset_info": self._get_dataset_info(dataset)
        }
        if group_metrics:
            results["group_metrics"] = {
                "group_column": self.distance_group_col,
                "metrics_by_group": group_metrics,
            }
        
        if self.save_detailed:
            results["detailed_scores"] = detailed_data
        
        # 保存结果
        if output_dir is not None:
            self._save_results(results, detailed_data, output_dir)
        
        # 打印主要指标
        self._print_classification_metrics(metrics)
        
        return results
    
    def _evaluate_legacy(self, 
                        model, 
                        dataset, 
                        output_dir: Optional[str],
                        progress_bar: bool) -> Dict[str, Any]:
        """
        兼容旧版本数据集的评估方式（基于类名判断）
        """
        dataset_type = type(dataset).__name__
        
        if "Dms" in dataset_type or "DMS" in dataset_type:
            # 假设是回归任务
            return self._evaluate_regression(model, dataset, output_dir, progress_bar)
        elif any(keyword in dataset_type for keyword in ["Genomic", "Variant", "TraitGym"]):
            # 假设是分类任务
            return self._evaluate_classification(model, dataset, output_dir, progress_bar)
        else:
            # 默认使用回归评估
            print(f"Warning: Unknown dataset type {dataset_type}, defaulting to regression evaluation")
            return self._evaluate_regression(model, dataset, output_dir, progress_bar)

    def _get_group_labels(self, dataset) -> Optional[List]:
        if not self.group_by_distance:
            return None
        if not hasattr(dataset, "df") or dataset.df is None:
            print("[WARN] group_by_distance enabled but dataset.df is missing; skip grouping.")
            return None
        if self.distance_group_col not in dataset.df.columns:
            print(f"[WARN] group_by_distance enabled but column '{self.distance_group_col}' not found; skip grouping.")
            return None
        groups = dataset.df[self.distance_group_col].tolist()
        if len(groups) != len(dataset):
            print("[WARN] group_by_distance length mismatch; skip grouping.")
            return None
        return groups

    def _compute_grouped_classification_metrics(
        self,
        delta_scores: np.ndarray,
        targets: np.ndarray,
        groups: List,
    ) -> Dict[str, Dict[str, float]]:
        if len(groups) != len(delta_scores) or len(groups) != len(targets):
            return {}

        group_metrics: Dict[str, Dict[str, float]] = {}
        groups_arr = np.asarray(groups, dtype=object)
        unique_groups = pd.unique(groups_arr)

        for group in unique_groups:
            if group is None or pd.isna(group):
                continue
            mask = groups_arr == group
            if not np.any(mask):
                continue
            metrics = self._compute_classification_metrics(delta_scores[mask], targets[mask])
            group_metrics[str(group)] = metrics

        return group_metrics
    
    def _compute_regression_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """计算回归任务指标"""
        # 检查数据有效性
        if len(predictions) != len(targets):
            raise ValueError("Predictions and targets must have the same length")
        
        # 移除 NaN 值
        valid_mask = ~(np.isnan(predictions) | np.isnan(targets))
        predictions = predictions[valid_mask]
        targets = targets[valid_mask]
        
        if len(predictions) == 0:
            raise ValueError("No valid predictions after removing NaN values")
        
        # 计算相关性
        spearman_corr, spearman_p = spearmanr(targets, predictions)
        pearson_corr, pearson_p = pearsonr(targets, predictions)
        
        return {
            "spearman_corr": float(spearman_corr),
            "spearman_p": float(spearman_p),
            "pearson_corr": float(pearson_corr),
            "pearson_p": float(pearson_p),
            "n_samples": len(targets),
            "n_valid_samples": len(predictions),
            "prediction_mean": float(np.mean(predictions)),
            "prediction_std": float(np.std(predictions)),
            "target_mean": float(np.mean(targets)),
            "target_std": float(np.std(targets))
        }
    
    def _compute_classification_metrics(self, delta_scores: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """计算分类任务指标"""
        # 检查数据有效性
        if len(delta_scores) != len(targets):
            raise ValueError("Delta scores and targets must have the same length")
        
        # 移除 NaN 值
        valid_mask = ~np.isnan(delta_scores)
        delta_scores = delta_scores[valid_mask]
        targets = targets[valid_mask]
        
        if len(delta_scores) == 0:
            raise ValueError("No valid delta scores after removing NaN values")
        
        # 基础统计指标
        metrics = {
            "n_samples": len(targets),
            "n_valid_samples": len(delta_scores),
            "delta_score_mean": float(np.mean(delta_scores)),
            "delta_score_std": float(np.std(delta_scores)),
            "target_mean": float(np.mean(targets)),
            "positive_ratio": float(np.mean(targets))
        }
        
        # 检查是否为二分类任务
        unique_targets = np.unique(targets)
        if len(unique_targets) == 2 and set(unique_targets).issubset({0, 1}):
            # 二分类指标
            try:
                # AUROC - 使用负的 delta_score（负变化表示有害）
                auroc = roc_auc_score(targets, -delta_scores)
                metrics["auroc"] = float(auroc)
                
                # AUPRC - 使用负的delta_score（负变化表示有害）
                auprc = average_precision_score(targets, -delta_scores)
                metrics["auprc"] = float(auprc)

                # 使用 delta_score 的阈值进行二分类预测
                # 负的 delta_score 预测为正类（有害）
                predictions = (delta_scores < 0).astype(int)
                
                metrics.update({
                    "accuracy": float(accuracy_score(targets, predictions)),
                    "precision": float(precision_score(targets, predictions, zero_division=0)),
                    "recall": float(recall_score(targets, predictions, zero_division=0)),
                    "f1_score": float(f1_score(targets, predictions, zero_division=0))
                })
                
            except Exception as e:
                print(f"Warning: Error computing classification metrics: {e}")
        
        # 相关性指标（总是计算）
        try:
            spearman_corr, spearman_p = spearmanr(targets, delta_scores)
            pearson_corr, pearson_p = pearsonr(targets, delta_scores)
            
            metrics.update({
                "spearman_corr": float(spearman_corr),
                "spearman_p": float(spearman_p),
                "pearson_corr": float(pearson_corr),
                "pearson_p": float(pearson_p)
            })
        except Exception as e:
            print(f"Warning: Error computing correlation metrics: {e}")
        
        return metrics
    
    def _get_model_info(self, model) -> Dict[str, Any]:
        """获取模型信息"""
        info = {"model_class": type(model).__name__}
        
        if hasattr(model, 'model_name'):
            info["model_name"] = model.model_name
        if hasattr(model, 'model_path'):
            info["model_path"] = model.model_path
        if hasattr(model, 'get_model_info'):
            info.update(model.get_model_info())
            
        return info
    
    # def _get_dataset_info(self, dataset) -> Dict[str, Any]:
    #     """获取数据集信息"""
    #     info = {
    #         "dataset_class": type(dataset).__name__,
    #         "dataset_size": len(dataset)
    #     }
        
    #     if hasattr(dataset, 'get_dataset_info'):
    #         info.update(dataset.get_dataset_info())
    #     if hasattr(dataset, 'get_task_type'):
    #         info["task_type"] = str(dataset.get_task_type())
            
    #     return info
    
    def _save_results(self, results: Dict[str, Any], detailed_data: List[Dict], output_dir: str):
        """保存评估结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存指标
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 保存详细结果
        if detailed_data:
            scores_path = os.path.join(output_dir, "detailed_scores.tsv")
            columns = detailed_data[0].keys()
            with open(scores_path, 'w') as f:
                f.write('\t'.join(columns) + '\n')
                for row in detailed_data:
                    values = [str(row.get(col, '')) for col in columns]
                    f.write('\t'.join(values) + '\n')
    
    def _print_regression_metrics(self, metrics: Dict[str, float]):
        """打印回归任务指标"""
        print(f"Regression Evaluation Results:")
        print(f"  Spearman correlation: {metrics['spearman_corr']:.4f} (p={metrics['spearman_p']:.4g})")
        print(f"  Pearson correlation: {metrics['pearson_corr']:.4f} (p={metrics['pearson_p']:.4g})")
        print(f"  Samples: {metrics['n_valid_samples']}/{metrics['n_samples']}")
    
    def _print_classification_metrics(self, metrics: Dict[str, float]):
        """打印分类任务指标"""
        print(f"Classification Evaluation Results:")
        if 'auroc' in metrics:
            print(f"  AUROC: {metrics['auroc']:.4f}")
        if 'auprc' in metrics:
            print(f"  AUPRC: {metrics['auprc']:.4f}")
        if 'accuracy' in metrics:
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
        if 'f1_score' in metrics:
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
        if 'spearman_corr' in metrics:
            print(f"  Spearman correlation: {metrics['spearman_corr']:.4f} (p={metrics['spearman_p']:.4g})")
        print(f"  Samples: {metrics['n_valid_samples']}/{metrics['n_samples']}")
        print(f"  Positive ratio: {metrics['positive_ratio']:.4f}")
