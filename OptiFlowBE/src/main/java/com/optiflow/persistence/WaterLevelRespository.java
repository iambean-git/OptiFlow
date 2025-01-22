package com.optiflow.persistence;


import java.time.LocalDateTime;
import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.optiflow.domain.WaterLevel;

@Repository
public interface WaterLevelRespository extends JpaRepository<WaterLevel, LocalDateTime> {
	List<WaterLevel> findByWaterLevelDt(LocalDateTime datetime);
}
