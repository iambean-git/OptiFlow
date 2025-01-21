package com.optiflow.persistence;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.optiflow.domain.Reservoir;

@Repository
public interface ReservoirRepository extends JpaRepository<Reservoir, Integer>{
}
