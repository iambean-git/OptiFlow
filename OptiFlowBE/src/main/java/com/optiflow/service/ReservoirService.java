package com.optiflow.service;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.optiflow.domain.Reservoir;
import com.optiflow.persistence.ReservoirRepository;

@Service
public class ReservoirService {
	
	@Autowired
	private ReservoirRepository reservoirRepo;
	
	public List<Reservoir> getAllReservoir(){
		return reservoirRepo.findAll();
	}
}
