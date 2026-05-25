//
// Created by chris on 1/28/26.
//
vk::Semaphore timeline_a = /* create timeline */;
vk::Semaphore timeline_b = /* create timeline */;
uint64_t	  counter_a	 = 0;
uint64_t	  counter_b	 = 0;

// Submit A - no waits, signals timeline_a to 1
{
	counter_a++;
	vk::SemaphoreSubmitInfo signal{
		.semaphore = timeline_a,
		.value	   = counter_a,
		.stageMask = vk::PipelineStageFlagBits2::eComputeShader,
	};
	vk::CommandBufferSubmitInfo cmd_info{.commandBuffer = cmd_a};

	queue.submit2(vk::SubmitInfo2{
		.commandBufferInfoCount	  = 1,
		.pCommandBufferInfos	  = &cmd_info,
		.signalSemaphoreInfoCount = 1,
		.pSignalSemaphoreInfos	  = &signal,
	});
}

// Submit B - no waits, signals timeline_b to 1 (runs parallel with A!)
{
	counter_b++;
	vk::SemaphoreSubmitInfo signal{
		.semaphore = timeline_b,
		.value	   = counter_b,
		.stageMask = vk::PipelineStageFlagBits2::eComputeShader,
	};
	vk::CommandBufferSubmitInfo cmd_info{.commandBuffer = cmd_b};

	queue.submit2(vk::SubmitInfo2{
		.commandBufferInfoCount	  = 1,
		.pCommandBufferInfos	  = &cmd_info,
		.signalSemaphoreInfoCount = 1,
		.pSignalSemaphoreInfos	  = &signal,
	});
}

// Submit C - waits on BOTH A and B, then signals its own timeline
{
	vk::Semaphore timeline_c = /* ... */;
	uint64_t	  counter_c	 = 1;

	std::array<vk::SemaphoreSubmitInfo, 2> waits{{
		{
			.semaphore = timeline_a,
			.value	   = counter_a,
			.stageMask = vk::PipelineStageFlagBits2::eComputeShader,
		},
		{
			.semaphore = timeline_b,
			.value	   = counter_b,
			.stageMask = vk::PipelineStageFlagBits2::eComputeShader,
		},
	}};

	vk::SemaphoreSubmitInfo signal{
		.semaphore = timeline_c,
		.value	   = counter_c,
		.stageMask = vk::PipelineStageFlagBits2::eComputeShader,
	};

	vk::CommandBufferSubmitInfo cmd_info{.commandBuffer = cmd_c};

	queue.submit2(vk::SubmitInfo2{
		.waitSemaphoreInfoCount	  = 2,
		.pWaitSemaphoreInfos	  = waits.data(),
		.commandBufferInfoCount	  = 1,
		.pCommandBufferInfos	  = &cmd_info,
		.signalSemaphoreInfoCount = 1,
		.pSignalSemaphoreInfos	  = &signal,
	});
}

device.waitSemaphores(
	{
		.semaphoreCount = 1,
		.pSemaphores	= &timeline_c,
		.pValues		= &counter_c,
	},
	UINT64_MAX);
// Now safe to map and read C's output buffer