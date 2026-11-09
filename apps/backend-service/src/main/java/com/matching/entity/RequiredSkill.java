package com.matching.entity;

import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.CreationTimestamp;

import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.util.UUID;

@Entity
@Table(name = "required_skills")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class RequiredSkill {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private UUID id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "jd_id", nullable = false)
    private JobDescription jobDescription;

    @Column(name = "skill_name", nullable = false, length = 100)
    private String skillName;

    @Column(nullable = false, length = 50)
    private String category;

    @Column(nullable = false, length = 20)
    @Builder.Default
    private String importance = "MANDATORY";

    @Column(nullable = false, precision = 3, scale = 2)
    @Builder.Default
    private BigDecimal weight = BigDecimal.valueOf(1.00);

    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private OffsetDateTime createdAt;
}
