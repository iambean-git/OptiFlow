package com.optiflow.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;

import com.optiflow.domain.Reservoir;
import com.optiflow.service.ReservoirService;

@Controller
@RequestMapping("/api/reservoir")
public class ReservoirController {
	
	@Autowired
	private ReservoirService reservoirService;
	
	@GetMapping
	public Reservoir getReservoirByName(@PathVariable String name) {
		return null;
	}
}
