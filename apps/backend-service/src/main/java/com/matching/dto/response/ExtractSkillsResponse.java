package com.matching.dto.response;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ExtractSkillsResponse {
    private List<SkillItemDto> skills;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class SkillItemDto {
        @JsonProperty("skill_name")
        private String skillName;

        private String category;
        private String importance;
        private Double weight;
    }
}
