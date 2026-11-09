package com.matching.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import com.matching.client.AiEngineClient;
import com.matching.dto.request.ExtractSkillsRequest;
import com.matching.dto.response.ApiResponse;
import com.matching.dto.response.ExtractSkillsResponse;
import com.matching.entity.JobDescription;
import com.matching.entity.RequiredSkill;
import com.matching.repository.JobDescriptionRepository;

import java.math.BigDecimal;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class JobService {

    private final JobDescriptionRepository jobDescriptionRepository;
    private final AiEngineClient aiEngineClient;

    @Transactional
    public ApiResponse<ExtractSkillsResponse> extractAndSaveSkills(ExtractSkillsRequest request) {
        log.info("Xử lý nghiệp vụ trích xuất kỹ năng JD: {}", request.getJobTitle());

        // 1. Lưu bản ghi cơ sở vào CSDL (PostgreSQL)
        JobDescription jobDescription = JobDescription.builder()
                .title(request.getJobTitle())
                .domain("BACKEND")
                .rawContent(request.getRawJdText())
                .build();
        JobDescription savedJob = jobDescriptionRepository.save(jobDescription);

        // 2. Điều phối gửi yêu cầu sang AI Engine (FastAPI)
        ApiResponse<ExtractSkillsResponse> aiResponse = aiEngineClient.extractSkills(request).block();

        // 3. Nếu AI trích xuất thành công, lưu các kỹ năng vào bảng required_skills
        if (aiResponse != null && aiResponse.getData() != null && aiResponse.getData().getSkills() != null) {
            var skills = aiResponse.getData().getSkills().stream().map(s -> RequiredSkill.builder()
                    .jobDescription(savedJob)
                    .skillName(s.getSkillName())
                    .category(s.getCategory() != null ? s.getCategory() : "GENERAL")
                    .importance(s.getImportance() != null ? s.getImportance() : "MANDATORY")
                    .weight(s.getWeight() != null ? BigDecimal.valueOf(s.getWeight()) : BigDecimal.ONE)
                    .build()).collect(Collectors.toList());
            savedJob.setRequiredSkills(skills);
            jobDescriptionRepository.save(savedJob);
        }

        return aiResponse;
    }
}
