package com.optiflow.controller;

import java.util.List;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.optiflow.domain.ReservoirData;
import com.optiflow.dto.ReservoirStats;
import com.optiflow.service.ReservoirDataService;

@RestController
@RequestMapping("/api/reservoirdata")
public class ReservoirDataController {

	@Autowired
	private ReservoirDataService reservoirDataService;

	@GetMapping
	public ResponseEntity<List<ReservoirData>> getAllReservoirData() {
		List<ReservoirData> datas = reservoirDataService.getAllReservoirData();
		return ResponseEntity.ok(datas);
	}

	@GetMapping("/daily/{reservoirId}")
	public ResponseEntity<List<ReservoirStats>> getDailyStatsByReservoirId(@PathVariable int reservoirId) {
		List<ReservoirStats> stats = reservoirDataService.findDailyStatsByReservoirId(reservoirId);
		return new ResponseEntity<>(stats, HttpStatus.OK);
	}
	
	@GetMapping("/monthly/{reservoirId}")
	public ResponseEntity<List<ReservoirStats>> getMonthlyStatsByReservoirId(@PathVariable int reservoirId) {
		List<ReservoirStats> stats = reservoirDataService.findMonthlyStatsByReservoirId(reservoirId);
		return new ResponseEntity<>(stats, HttpStatus.OK);
	}
//	@GetMapping("/monthly/{reservoirId}")
//	public ResponseEntity<List<ReservoirStats>> getMonthlyStatsByReservoirId(@PathVariable Long reservoirId) {
//		List<ReservoirStats> stats = reservoirDataService.getMonthlyStatsByReservoirId(reservoirId);
//		return new ResponseEntity<>(stats, HttpStatus.OK);
//	}
//
//	@GetMapping("/yearly/{reservoirId}")
//	public ResponseEntity<List<ReservoirStats>> getYearlyStatsByReservoirId(@PathVariable Long reservoirId) {
//		List<ReservoirStats> stats = reservoirDataService.getYearlyStatsByReservoirId(reservoirId);
//		return new ResponseEntity<>(stats, HttpStatus.OK);
//	}
//	@GetMapping("/stats")
//    public ResponseEntity<List<Object>> getReservoirStats(
//            @RequestParam("groupBy") String groupBy) {
//        return ResponseEntity.ok(reservoirDataService.getReservoirStatsByGroup(groupBy));
//    }
	
//	@GetMapping("/stats")
//	public ResponseEntity<List<ReservoirStats>> getReservoirStats(int year) {
//      return ResponseEntity.ok(reservoirDataService.findByYear(year));
//  }
}
