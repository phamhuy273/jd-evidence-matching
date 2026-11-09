package com.matching.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import com.matching.entity.RequiredSkill;

import java.util.List;
import java.util.UUID;

@Repository
public interface RequiredSkillRepository extends JpaRepository<RequiredSkill, UUID> {
    List<RequiredSkill> findByJobDescriptionId(UUID jdId);
}
