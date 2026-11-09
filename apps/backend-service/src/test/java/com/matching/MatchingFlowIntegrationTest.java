package com.matching;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import reactor.core.publisher.Mono;
import com.matching.client.AiEngineClient;
import com.matching.dto.request.EvaluateAsyncRequest;
import com.matching.dto.request.ExtractSkillsRequest;
import com.matching.dto.response.*;

import java.util.List;
import java.util.UUID;

import static org.hamcrest.Matchers.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@SpringBootTest
@AutoConfigureMockMvc
public class MatchingFlowIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private AiEngineClient aiEngineClient;

    @Test
    @DisplayName("Test 1: Kích hoạt đối sánh bất đồng bộ nhận mã 202 Accepted (Mục 4.3)")
    void testEvaluateAsyncFlow() throws Exception {
        UUID jobId = UUID.randomUUID();
        String taskId = UUID.randomUUID().toString();

        EvaluateAsyncResponse mockResponse = EvaluateAsyncResponse.builder()
                .taskId(taskId)
                .status("PROCESSING")
                .isCached(false)
                .build();

        Mockito.when(aiEngineClient.evaluateAsync(Mockito.any(EvaluateAsyncRequest.class)))
                .thenReturn(Mono.just(ApiResponse.accepted("Đã tiếp nhận yêu cầu phân tích.", mockResponse)));

        EvaluateAsyncRequest request = EvaluateAsyncRequest.builder()
                .jobId(jobId)
                .candidateName("Nguyễn Văn A")
                .githubRepoUrl("https://github.com/nguyenvana/spring-ecommerce-api")
                .commitHash("7b8f9e0123456789abcdef0123456789abcdef01")
                .build();

        mockMvc.perform(post("/api/v1/matching/evaluate-async")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.code", is(202)))
                .andExpect(jsonPath("$.status", is("ACCEPTED")))
                .andExpect(jsonPath("$.data.task_id", is(taskId)))
                .andExpect(jsonPath("$.data.status", is("PROCESSING")));
    }

    @Test
    @DisplayName("Test 2: Polling trạng thái tác vụ ngầm nhận mã 200 OK (Mục 4.4 Endpoint 1)")
    void testPollingTaskStatusFlow() throws Exception {
        String taskId = UUID.randomUUID().toString();

        TaskStatusResponse mockResponse = TaskStatusResponse.builder()
                .taskId(taskId)
                .progressStatus("COMPLETED")
                .progressPercent(100)
                .build();

        Mockito.when(aiEngineClient.getTaskStatus(taskId))
                .thenReturn(Mono.just(ApiResponse.success(mockResponse)));

        mockMvc.perform(get("/api/v1/matching/tasks/{task_id}/status", taskId)
                        .accept(MediaType.APPLICATION_JSON))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code", is(200)))
                .andExpect(jsonPath("$.status", is("SUCCESS")))
                .andExpect(jsonPath("$.data.task_id", is(taskId)))
                .andExpect(jsonPath("$.data.progress_status", is("COMPLETED")))
                .andExpect(jsonPath("$.data.progress_percent", is(100)));
    }

    @Test
    @DisplayName("Test 3: Truy vấn danh sách xếp hạng ứng viên và minh chứng code (Mục 4.4 Endpoint 2)")
    void testGetJobResultsFlow() throws Exception {
        UUID jobId = UUID.randomUUID();

        JobResultsResponse mockResponse = JobResultsResponse.builder()
                .jobId(jobId)
                .ranking(List.of(
                        JobResultsResponse.CandidateRankingDto.builder()
                                .rank(1)
                                .candidateName("Nguyễn Văn A")
                                .totalScore(88.5)
                                .faithfulnessScore(0.92)
                                .skillsAssessment(List.of(
                                        JobResultsResponse.SkillAssessmentDto.builder()
                                                .skillName("Spring Boot")
                                                .score(95.0)
                                                .evidenceCodeChunk(JobResultsResponse.EvidenceCodeChunkDto.builder()
                                                        .filePath("src/main/java/com/app/controller/OrderController.java")
                                                        .codeSnippet("@PostMapping(\"/orders\")...")
                                                        .build())
                                                .llmExplanation("Hiện thực đầy đủ REST Controller.")
                                                .build()
                                ))
                                .build()
                ))
                .build();

        Mockito.when(aiEngineClient.getJobResults(jobId))
                .thenReturn(Mono.just(ApiResponse.success(mockResponse)));

        mockMvc.perform(get("/api/v1/matching/jobs/{job_id}/results", jobId)
                        .accept(MediaType.APPLICATION_JSON))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code", is(200)))
                .andExpect(jsonPath("$.status", is("SUCCESS")))
                .andExpect(jsonPath("$.data.ranking[0].candidate_name", is("Nguyễn Văn A")))
                .andExpect(jsonPath("$.data.ranking[0].skills_assessment[0].skill_name", is("Spring Boot")));
    }

    @Test
    @DisplayName("Test 4: Bóc tách kỹ năng từ JD và lưu vào CSDL (Mục 4.2)")
    void testExtractSkillsFlow() throws Exception {
        ExtractSkillsRequest request = ExtractSkillsRequest.builder()
                .jobTitle("Backend Java Software Engineer")
                .rawJdText("Yêu cầu thành thạo Java 17, Spring Boot, JPA, Docker.")
                .build();

        ExtractSkillsResponse mockResponse = ExtractSkillsResponse.builder()
                .skills(List.of(
                        ExtractSkillsResponse.SkillItemDto.builder()
                                .skillName("Java")
                                .category("LANGUAGE")
                                .importance("MANDATORY")
                                .weight(0.3)
                                .build(),
                        ExtractSkillsResponse.SkillItemDto.builder()
                                .skillName("Spring Boot")
                                .category("FRAMEWORK")
                                .importance("MANDATORY")
                                .weight(0.3)
                                .build()
                ))
                .build();

        Mockito.when(aiEngineClient.extractSkills(Mockito.any(ExtractSkillsRequest.class)))
                .thenReturn(Mono.just(ApiResponse.success(mockResponse)));

        mockMvc.perform(post("/api/v1/jobs/extract-skills")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.code", is(200)))
                .andExpect(jsonPath("$.data.skills[0].skill_name", is("Java")))
                .andExpect(jsonPath("$.data.skills[1].skill_name", is("Spring Boot")));
    }

    @Test
    @DisplayName("Test 5: Validation xử lý lỗi INVALID_INPUT_DATA (400)")
    void testValidationErrorFlow() throws Exception {
        ExtractSkillsRequest badRequest = ExtractSkillsRequest.builder()
                .jobTitle("") // Trống tiêu đề
                .rawJdText("ngắn") // Dưới 10 ký tự
                .build();

        mockMvc.perform(post("/api/v1/jobs/extract-skills")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(badRequest)))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code", is(400)))
                .andExpect(jsonPath("$.status", is("INVALID_INPUT_DATA")));
    }
}
