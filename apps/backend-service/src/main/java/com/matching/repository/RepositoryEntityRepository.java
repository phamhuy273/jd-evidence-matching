package com.matching.repository;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;
import com.matching.entity.RepositoryEntity;

import java.util.Optional;
import java.util.UUID;

@Repository
public interface RepositoryEntityRepository extends JpaRepository<RepositoryEntity, UUID> {
    Optional<RepositoryEntity> findByRepoUrlAndCommitHash(String repoUrl, String commitHash);
}
