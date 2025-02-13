package com.optiflow.service;

import java.util.List;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import com.optiflow.domain.Member;
import com.optiflow.persistence.MemberRepository;

@Service
public class MemberService {
	
	@Autowired
	private MemberRepository memberRepo;
	
	@Autowired
	private PasswordEncoder encoder;
	

	public Member saveMember(Member member) {
    	Member memberRegi = Member.builder()
                .username(member.getUsername())
                .password(encoder.encode(member.getPassword()))
                .role(Member.Role.Role_User)
                .build();
		
		return memberRepo.save(memberRegi);
	}

	public List<Member> getAllMembers() {
		return memberRepo.findAll();
	}

	public Optional<Member> getMemberByUsername(String username) {
		return memberRepo.findByUsername(username);
	}

	public void deleteMember(String username) {
		memberRepo.deleteByUsername(username);
	}
}
