package com.optiflow.persistence;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.optiflow.domain.Reservoir;

@Repository
public interface ReservoirRepository extends JpaRepository<Reservoir, Integer>{
	
	Optional<Reservoir> findByName(String name);
}
