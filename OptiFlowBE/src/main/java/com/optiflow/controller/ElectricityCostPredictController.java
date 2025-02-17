package com.optiflow.controller;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.optiflow.domain.Reservoir;
import com.optiflow.dto.ElectricityCostPredictRequestDto;
import com.optiflow.dto.ElectricityCostPredictResponseDto;
import com.optiflow.persistence.ReservoirDataRepository;
import com.optiflow.persistence.ReservoirRepository;
import com.optiflow.service.ElectricityCostPredictService;

import io.swagger.v3.oas.annotations.tags.Tag;

@RestController
@RequestMapping("/api")
@Tag(name = "ElectrictityCostPredict API", description = "전력량 예측 관련 API")
public class ElectricityCostPredictController {
	
	private static final Logger log = LoggerFactory.getLogger(WaterDemandPredictController.class);
	
	@Autowired
	private ReservoirRepository reservoirRepo;
	
	@Autowired
	private ReservoirDataRepository reservoirDataRepo;
	
	@Autowired
	private ElectricityCostPredictService costService;
	
	@GetMapping("/hourlycost/{reservoirName}/{datetime}")
    public ResponseEntity<Map<String, List<?>>> getPrediction(@PathVariable String reservoirName, @PathVariable String datetime) {
		log.info("Received prediction request with datetime: {}", datetime);
		Optional<Reservoir> reservoirOptional = reservoirRepo.findByName(reservoirName);
		Float reservoirArea = reservoirOptional.get().getArea();
		LocalDateTime localDateTime = LocalDateTime.parse(datetime);
		int reservoirId = reservoirOptional.get().getReservoirId();
		Float height = reservoirDataRepo.findHeightByReservoirIdAndObservationTime(reservoirId, localDateTime);
		Float waterLevel = reservoirArea * height;
		ElectricityCostPredictRequestDto requestDto = new ElectricityCostPredictRequestDto(); 
        
        requestDto.setName(reservoirName);
        requestDto.setDatetime(datetime);
        requestDto.setWaterLevel(waterLevel);
        ElectricityCostPredictResponseDto responseDto = costService.getPrediction(requestDto);
        List<ElectricityCostPredictResponseDto> datas = new ArrayList<>();
        datas.add(responseDto);

        Map<String, List<?>> responseMap = costService.convertToResponseMap(datas);        
        return ResponseEntity.ok(responseMap);
    }
	
	@GetMapping("/dailycost/{name}/{datetime}")
    public ResponseEntity<Map<String, List<?>>> getDailyCost(
            @PathVariable String name,
            @PathVariable String datetime) {
        Map<String, List<?>> responseMap = costService.getDailyCostData(name, datetime);
        return ResponseEntity.ok(responseMap);
    }
	
	@GetMapping("/monthlycost/{name}/{datetime}")
    public ResponseEntity<Map<String, List<?>>> getMonthlyCost(
            @PathVariable String name,
            @PathVariable String datetime) {
        Map<String, List<?>> responseMap = costService.getMonthlyCostData(name, datetime);
        return ResponseEntity.ok(responseMap);
    }
}
