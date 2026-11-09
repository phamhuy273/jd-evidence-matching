package com.matching.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.UUID;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class JobResultsResponse {

    @JsonProperty("job_id")
    private UUID jobId;

    private List<CandidateRankingDto> ranking;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class CandidateRankingDto {
        private Integer rank;

        @JsonProperty("candidate_name")
        private String candidateName;

        @JsonProperty("total_score")
        private Double totalScore;

        @JsonProperty("faithfulness_score")
        private Double faithfulnessScore;

        @JsonProperty("skills_assessment")
        private List<SkillAssessmentDto> skillsAssessment;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class SkillAssessmentDto {
        @JsonProperty("skill_name")
        private String skillName;

        private Double score;

        @JsonProperty("evidence_code_chunk")
        private EvidenceCodeChunkDto evidenceCodeChunk;

        @JsonProperty("llm_explanation")
        private String llmExplanation;
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class EvidenceCodeChunkDto {
        @JsonProperty("file_path")
        private String filePath;

        @JsonProperty("code_snippet")
        private String codeSnippet;
    }
}
