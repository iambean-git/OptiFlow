package com.optiflow.controller;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.optiflow.domain.Reservoir;
import com.optiflow.domain.ReservoirData;
import com.optiflow.dto.ReservoirStats;
import com.optiflow.persistence.ReservoirRepository;
import com.optiflow.service.ReservoirDataService;

@RestController
@RequestMapping("/api/reservoirdata")
public class ReservoirDataController {

	@Autowired
	private ReservoirDataService reservoirDataService;
	@Autowired
	private ReservoirRepository reservoirRepo;
	
	@GetMapping("/{reservoirName}/{datetime}")
	public ResponseEntity<?> findByObservationTime(@PathVariable String reservoirName,
			@PathVariable String datetime) {
		
		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(reservoirName);
		if (!reservoirOptional.isPresent()) { // 없을 경우 404 에러 반환
			return new ResponseEntity<>(HttpStatus.NOT_FOUND);
		}
		Reservoir reservoir = reservoirOptional.get();
		LocalDateTime localDateTime = LocalDateTime.parse(datetime);
		LocalDateTime startTime = localDateTime.minusHours(24);
		LocalDateTime endTime = localDateTime.minusHours(1); // 요청된 시간까지 포함
		
		List<ReservoirData> datas = reservoirDataService.findByObservationTimeRange(reservoir, startTime, endTime);
		if (datas.isEmpty()) {
	        return ResponseEntity.ok(Collections.emptyMap()); // 데이터가 없을 경우 빈 JSON 객체 반환 또는 다른 처리
	    }

	    Map<String, List<?>> responseMap = new HashMap<>();
	    List<String> timeList = new ArrayList<>();
	    List<Float> inputList = new ArrayList<>();
	    List<Float> outputList = new ArrayList<>();
	    List<Float> heightList = new ArrayList<>();

	    for (ReservoirData data : datas) {
	        timeList.add(data.getObservationTime().toString());
	        inputList.add(data.getInput());
	        outputList.add(data.getOutput());
	        heightList.add(data.getHeight() / reservoir.getHeight() * 100);
	    }

	    responseMap.put("time", timeList);
	    responseMap.put("input", inputList);
	    responseMap.put("output", outputList);
	    responseMap.put("height", heightList);

	    return ResponseEntity.ok(responseMap);
	}
	@GetMapping("/{datetime}")
	public ResponseEntity<List<ReservoirData>> findByObservationTime(@PathVariable String datetime){
		LocalDateTime localDateTime = LocalDateTime.parse(datetime);
		List<ReservoirData> datas = reservoirDataService.findByObservationTime(localDateTime);
		return ResponseEntity.ok(datas);
	}

	@GetMapping("/daily/{reservoirName}")
	public ResponseEntity<List<ReservoirStats>> getDailyStatsByReservoirId(@PathVariable String reservoirName) {
		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(reservoirName);
		
        if (!reservoirOptional.isPresent()) { // 없을 경우 404 에러 반환
            return new ResponseEntity<>(HttpStatus.NOT_FOUND); 
        }
        int reservoirId = reservoirOptional.get().getReservoirId();
		List<ReservoirStats> stats = reservoirDataService.findDailyStatsByReservoirId(reservoirId);
		return new ResponseEntity<>(stats, HttpStatus.OK);
	}

	 // 시간별 통계 (일별 기준)
    @GetMapping("/hourly/{date}/{reservoirName}") 
    public ResponseEntity<List<ReservoirStats>> findHourlyStatsByDailyObservationTimeAndReservoirId(
            @PathVariable String date,
            @PathVariable String reservoirName) {
    	
		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(reservoirName);
        if (!reservoirOptional.isPresent()) { 
            return new ResponseEntity<>(HttpStatus.NOT_FOUND); 
        }
        int reservoirId = reservoirOptional.get().getReservoirId();
        List<ReservoirStats> stats = reservoirDataService.findHourlyStatsByDailyObservationTimeAndReservoirId(date, reservoirId);
        return new ResponseEntity<>(stats, HttpStatus.OK);
    }

    // 일별 통계 (월별 기준)
    @GetMapping("/daily/{month}/{reservoirName}")
    public ResponseEntity<List<ReservoirStats>> findDailyStatsByMonthlyObservationTimeAndReservoirId(
            @PathVariable String month,
            @PathVariable String reservoirName) {
    	
		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(reservoirName);
        if (!reservoirOptional.isPresent()) {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
        int reservoirId = reservoirOptional.get().getReservoirId();
        List<ReservoirStats> stats = reservoirDataService.findDailyStatsByMonthlyObservationTimeAndReservoirId(month, reservoirId);
        return new ResponseEntity<>(stats, HttpStatus.OK);
    }

    // 월별 통계 (년별 기준)
    @GetMapping("/monthly/{year}/{reservoirName}") 
    public ResponseEntity<List<ReservoirStats>> findMonthlyStatsByYearlyObservationTimeAndReservoirId(
            @PathVariable String year,
            @PathVariable String reservoirName) {
    	
		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(reservoirName);
        if (!reservoirOptional.isPresent()) {
            return new ResponseEntity<>(HttpStatus.NOT_FOUND);
        }
        int reservoirId = reservoirOptional.get().getReservoirId();
        List<ReservoirStats> stats = reservoirDataService.findMonthlyStatsByYearlyObservationTimeAndReservoirId(year, reservoirId);
        return new ResponseEntity<>(stats, HttpStatus.OK);
    }
}
