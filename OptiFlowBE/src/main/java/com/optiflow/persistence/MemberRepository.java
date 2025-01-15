package com.optiflow.persistence;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.optiflow.domain.Member;

@Repository
public interface MemberRepository extends JpaRepository<Member, String> {

	Optional<Member> findByUsername(String username);

	void deleteByUsername(String username);

	boolean existsByUsername(String username);

}
