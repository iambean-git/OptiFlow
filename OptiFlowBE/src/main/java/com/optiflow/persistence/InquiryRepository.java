package com.optiflow.persistence;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.optiflow.domain.Inquiry;

@Repository
public interface InquiryRepository extends JpaRepository<Inquiry, Long> {
	List<Inquiry> findAllByStaffConfirmedIsFalse();
}
