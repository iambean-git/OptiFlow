package com.optiflow.controller;

import java.time.LocalDateTime;
import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.optiflow.domain.WaterLevel;
import com.optiflow.persistence.WaterLevelRespository;

@RestController
@RequestMapping("/api/waterlevels")
public class WaterLevelController {
	@Autowired
	private WaterLevelRespository waterLevelRepo;
	
	@GetMapping
	public List<WaterLevel> getAllWaterLevel() {
		return waterLevelRepo.findAll();
	}
	
	@GetMapping("/{datetime}")
	public List<WaterLevel> findByWaterLevelDt(@PathVariable String datetime){
		LocalDateTime localDateTime = LocalDateTime.parse(datetime);
		return waterLevelRepo.findByWaterLevelDt(localDateTime);
	}
}
