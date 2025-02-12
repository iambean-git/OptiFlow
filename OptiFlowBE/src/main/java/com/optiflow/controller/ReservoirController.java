package com.optiflow.controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.optiflow.domain.Reservoir;
import com.optiflow.service.ReservoirService;

import io.swagger.v3.oas.annotations.tags.Tag;

@RestController
@RequestMapping("/api")
@Tag(name = "Reservoir API", description = "배수지 관련 API")
public class ReservoirController {
	
	@Autowired
	private ReservoirService reservoirService;
	
	@GetMapping("/reservoirs")
	public List<Reservoir> getAllReservoir() {
		return reservoirService.getAllReservoir();
	}
}
