package know.engine.eval.metrics;

import know.engine.eval.model.EvalCase;
import know.engine.eval.model.RetrievalScores;
import org.springframework.stereotype.Component;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * 检索层指标计算器：Hit@K、MRR、Recall@K
 */
@Component
public class RetrievalMetricCalculator {

    // 按 Ground Truth 计算检索指标
    public RetrievalScores calculate(List<String> retrievedChunkIds,
                                     List<String> retrievedDocumentIds,
                                     EvalCase evalCase,
                                     int topK) {
        // 分片级 Ground Truth 集合
        Set<String> gtChunks = toSet(evalCase.getGroundTruthChunkIds());
        // 文档级 Ground Truth 集合
        Set<String> gtDocs = toSet(evalCase.getGroundTruthDocumentIds());

        // 截取 TopK 分片
        List<String> topChunks = sliceTop(retrievedChunkIds, topK);
        // 截取 TopK 文档
        List<String> topDocs = sliceTop(retrievedDocumentIds, topK);

        // 优先使用分片级评测
        boolean useChunkLevel = !gtChunks.isEmpty();
        // 无分片 GT 时回退到文档级
        boolean useDocLevel = !useChunkLevel && !gtDocs.isEmpty();

        // 两类 GT 都没有则返回未评分结果
        if (!useChunkLevel && !useDocLevel) {
            // hit=false，MRR/Recall 为 0
            return new RetrievalScores(false, 0.0, 0.0);
        }

        // 计算 Hit@K：TopK 是否命中任一 GT
        boolean hitAtK = useChunkLevel
                ? topChunks.stream().anyMatch(gtChunks::contains)
                : topDocs.stream().anyMatch(gtDocs::contains);

        // 计算 MRR
        double mrr = useChunkLevel
                ? computeMrr(topChunks, gtChunks)
                : computeMrr(topDocs, gtDocs);

        // 计算 Recall@K
        double recallAtK = useChunkLevel
                ? computeRecall(topChunks, gtChunks)
                : computeRecall(topDocs, gtDocs);

        // 返回三项指标
        return new RetrievalScores(hitAtK, mrr, recallAtK);
    }

    // 计算 Mean Reciprocal Rank
    private double computeMrr(List<String> retrieved, Set<String> groundTruth) {
        // 按排名扫描首次命中
        for (int i = 0; i < retrieved.size(); i++) {
            // 命中 Ground Truth
            if (groundTruth.contains(retrieved.get(i))) {
                // 返回 1/排名
                return 1.0 / (i + 1);
            }
        }
        // 未命中
        return 0.0;
    }

    // 计算 Recall@K
    private double computeRecall(List<String> retrieved, Set<String> groundTruth) {
        // 统计命中的不重复 GT 数
        long hitCount = retrieved.stream().filter(groundTruth::contains).distinct().count();
        // 命中数 / GT 总数
        return (double) hitCount / groundTruth.size();
    }

    // 截取列表前 topK 个元素
    private List<String> sliceTop(List<String> ids, int topK) {
        if (ids == null || ids.isEmpty()) {
            return List.of();
        }
        // 截取不超过列表长度的前 K 项
        return ids.subList(0, Math.min(topK, ids.size()));
    }

    // 将列表转为 Set，便于包含判断
    private Set<String> toSet(List<String> values) {
        if (values == null || values.isEmpty()) {
            return Set.of();
        }
        return new HashSet<>(values);
    }
}
