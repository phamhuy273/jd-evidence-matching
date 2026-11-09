package com.matching.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import com.matching.entity.EvaluationResult;

import java.util.List;
import java.util.UUID;

@Repository
public interface EvaluationResultRepository extends JpaRepository<EvaluationResult, UUID> {
    List<EvaluationResult> findByJobDescriptionIdOrderByTotalScoreDesc(UUID jdId);
}
