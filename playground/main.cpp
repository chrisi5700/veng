#include <atomic>
#include <functional>
#include <memory>
#include <print>
#include <thread>
#include <vector>

enum class State
{
	Valid,
	Processing,
	Invalid,
};

struct CPUNode;

struct Data
{
	virtual CPUNode*			  get_parent() = 0;
	virtual void			  invalidate() = 0;
	[[nodiscard]] const std::atomic<State>& request();
	bool					  valid();

	virtual ~Data() = default;
};

struct CPUNode
{
	 private:
	std::atomic<State> m_state = State::Invalid;

	 protected:
	virtual void invalidate_output() = 0;
	virtual void schedule(std::function<void()> f) { std::thread(f).detach(); }
	virtual std::vector<const std::atomic<State>*> get_pending_input() = 0;
	virtual void								   execute()		   = 0;

	 public:
	bool valid() const { return m_state == State::Valid; }
	void invalidate()
	{
		m_state.store(State::Invalid, std::memory_order_relaxed);
		invalidate_output();
	}
	std::atomic<State>& process()
	{
		auto expected = State::Invalid;
		if (!m_state.compare_exchange_strong(expected, State::Processing, std::memory_order_acquire,
										   std::memory_order_relaxed))
		{
			return m_state;
		}

		for (auto* state : get_pending_input())
		{
			state->wait(State::Processing, std::memory_order_acquire);
		}

		schedule(
			[&]()
			{
				execute();
				m_state.store(State::Valid, std::memory_order_release);
				m_state.notify_all();
			});
		return m_state;
	}

	virtual ~CPUNode() = default;
};

const std::atomic<State>& Data::request()
{
	auto parent = get_parent();
	if (not parent)
		throw std::logic_error{"No parent node"};
	return parent->process();
}

bool Data::valid()
{
	auto parent = get_parent();
	if (not parent)
		return true;
	return parent->valid();
}

template <class T>
class CPUData : public Data
{
	T				   held;
	std::vector<CPUNode*> consumers;
	CPUNode*			   parent;

	 public:
	explicit CPUData(const T& held, CPUNode* parent = nullptr)
		: held(held)
		, parent(parent)
	{
	}

	CPUNode* get_parent() override { return parent; }
	void  invalidate() override
	{
		for (auto consumer : consumers)
		{
			consumer->invalidate();
		}
	}

	void add_consumer(CPUNode* consumer) { consumers.push_back(consumer); }
	void set_data(const T& new_held)
	{
		held = new_held;
	}
	void update_data(const T& new_held)
	{
		held = new_held;
		invalidate();
	}
	const T& get_data() { return held; }
	~CPUData() override = default;
};

class IFSNode : public CPUNode
{
	CPUData<std::size_t>*	   particle_count = nullptr;
	CPUData<std::string>*	   ifs_name		  = nullptr;
	CPUData<std::vector<int>>* particles	  = nullptr;

	 protected:
	std::vector<const std::atomic<State>*> get_pending_input() override
	{
		std::vector<const std::atomic<State>*> input_states;

		if (not particle_count->valid())
		{
			std::println("IFSNode particles count changed requesting new generation");
			input_states.push_back(&particle_count->request());
		}
		if (not ifs_name->valid())
		{
			std::println("IFSNode name changed requesting new generation");
			input_states.push_back(&ifs_name->request());
		}
		return input_states;
	}

	void execute() override
	{
		std::println("IFSNode: Calculating IFS");

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		particles->set_data(std::vector<int>(particle_count->get_data()));
		std::println("IFSNode: Done calculating IFS");

	}
	void invalidate_output() override { particles->invalidate(); }

	 public:
	static std::pair<std::unique_ptr<IFSNode>, std::unique_ptr<CPUData<std::vector<int>>>> create(
		CPUData<std::size_t>* particle_count, CPUData<std::string>* ifs_name)
	{
		auto ifs			= std::make_unique<IFSNode>();
		auto out			= std::make_unique<CPUData<std::vector<int>>>(std::vector<int>{}, ifs.get());
		ifs->ifs_name		= ifs_name;
		ifs->particle_count = particle_count;
		ifs->particles		= out.get();
		particle_count->add_consumer(ifs.get());
		ifs_name->add_consumer(ifs.get());
		return std::make_pair(std::move(ifs), std::move(out));
	}

	~IFSNode() override = default;
};

template <class F>
class CPUTransformer;

template <class Ret, class... Args>
class CPUTransformer<Ret(Args...)> : public CPUNode
{
	CPUData<Ret>*									   output;
	std::tuple<CPUData<std::remove_cvref_t<Args>>*...> args;
	std::function<Ret(Args...)>						   func;

	 protected:
	void invalidate_output() override { output->invalidate(); }

	std::vector<const std::atomic<State>*> get_pending_input() override
	{
		std::vector<const std::atomic<State>*> input_states;

		[&]<std::size_t... Is>(std::index_sequence<Is...>)
		{
			(
				[&](auto* arg)
				{
					if (not arg->valid())
					{
						input_states.push_back(&arg->request());
					}
				}(std::get<Is>(args)),
				...);
		}(std::make_index_sequence<sizeof...(Args)>{});
		return input_states;
	}
	void execute() override
	{
		std::println("CPUTransformer: Executing stored function");
		auto data = [&]<std::size_t... Is>(std::index_sequence<Is...>)
		{
			return func([](auto* arg) { return arg->get_data(); }(std::get<Is>(args))...);
		}(std::make_index_sequence<sizeof...(Args)>{});
		output->set_data(data);
		std::println("CPUTransformer: Finished executing stored function");
	}

	 public:
	CPUTransformer(std::tuple<CPUData<std::remove_cvref_t<Args>>*...> args, std::function<Ret(Args...)> func)
		: output(nullptr)
		, args(args)
		, func(func)
	{
	}

	static std::pair<std::unique_ptr<CPUTransformer>, std::unique_ptr<CPUData<Ret>>> create(
		std::function<Ret(Args...)> func, CPUData<std::remove_cvref_t<Args>>*... data)
	{
		auto args	  = std::make_tuple(data...);
		auto trans	  = std::make_unique<CPUTransformer>(args, func);
		auto ret	  = std::make_unique<CPUData<Ret>>(Ret{}, trans.get());
		trans->output = ret.get();
		(data->add_consumer(trans.get()), ...);
		return std::make_pair(std::move(trans), std::move(ret));
	}


};

int main()
{
	CPUData<std::size_t> count{1024};
	CPUData<std::string> ifs_name{"Triangle"};
	auto [ifs, output]		   = IFSNode::create(&count, &ifs_name);
	auto [name_mapper, mapped] = CPUTransformer<std::vector<int>(const std::vector<int>&, const std::string&)>::create(
		[](const std::vector<int>& data, const std::string& name)
		{
			std::println("Mapping name...");
			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			auto out = data;
			for (auto& e : out)
			{
				e += name.length();
			}
			std::println("Finished mapping name");
			return out;
		},
		output.get(), &ifs_name);

	if (not mapped->valid())
	{
		auto& map_wait = mapped->request();
		map_wait.wait(State::Processing, std::memory_order_acquire);
		std::println("{}", mapped->get_data()[0]);
	}

	ifs_name.update_data("New Data Which is longer");

	if (not mapped->valid())
	{
		auto& map_wait = mapped->request();
		map_wait.wait(State::Processing, std::memory_order_acquire);
		std::println("{}", mapped->get_data()[0]);
	}
}