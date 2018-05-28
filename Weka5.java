package test.weka.source;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;

public class Weka5
{

	protected Classifier m_Classifier = null;
	protected Filter m_Filter = null;
	protected String m_TrainingFile = null;
	protected Instances m_Training = null;
	protected Evaluation m_Evaluation = null;

	public Weka5()
	{
		super();
	}

	public void setClassifier(String name, String[] options) throws Exception
	{
		Object obj = Class.forName(name).newInstance();
		m_Classifier = (Classifier) Class.forName(name).cast(obj);
	}

	public void setFilter(String name, String[] options) throws Exception
	{
		m_Filter = (Filter) Class.forName(name).newInstance();
		if (m_Filter instanceof OptionHandler)
			((OptionHandler) m_Filter).setOptions(options);
	}

	/**
	 * sets the file to use for training
	 */
	public void setTraining(String name) throws Exception
	{
		m_TrainingFile = name;
		m_Training = new Instances(new BufferedReader(new FileReader(m_TrainingFile)));
		m_Training.setClassIndex(m_Training.numAttributes() - 1);
	}

	/**
	 * runs 10fold CV over the training file
	 */
	public void execute() throws Exception
	{
		// run filter
		m_Filter.setInputFormat(m_Training);
		Instances filtered = Filter.useFilter(m_Training, m_Filter);

		// train classifier on complete file for tree
		m_Classifier.buildClassifier(filtered);

		// 10fold CV with seed=1
		m_Evaluation = new Evaluation(filtered);
		m_Evaluation.crossValidateModel(m_Classifier, filtered, 10, m_Training.getRandomNumberGenerator(1));
	}

	/**
	 * outputs some data about the classifier
	 */
	public String toString()
	{
		StringBuffer result;

		result = new StringBuffer();
		result.append("Weka - Demo\n===========\n\n");

		result.append("Classifier...: " + m_Classifier.getClass().getName() + " " + Utils.joinOptions(((OptionHandler) m_Classifier).getOptions()) + "\n");
		if (m_Filter instanceof OptionHandler)
			result.append("Filter.......: " + m_Filter.getClass().getName() + " " + Utils.joinOptions(((OptionHandler) m_Filter).getOptions()) + "\n");
		else
			result.append("Filter.......: " + m_Filter.getClass().getName() + "\n");
		result.append("Training file: " + m_TrainingFile + "\n");
		result.append("\n");

		result.append(m_Classifier.toString() + "\n");
		result.append(m_Evaluation.toSummaryString() + "\n");
		try
		{
			result.append(m_Evaluation.toMatrixString() + "\n");
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
		try
		{
			result.append(m_Evaluation.toClassDetailsString() + "\n");
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}

		return result.toString();
	}

	/**
	 * returns the usage of the class
	 */
	public static String usage()
	{
		return "\nusage:\n  " + Weka5.class.getName() + "  CLASSIFIER <classname> [options] \n" + "  FILTER <classname> [options]\n" + "  DATASET <trainingfile>\n\n" + "e.g., \n" + "  java -classpath \".:weka.jar\" Weka5 \n" + "    CLASSIFIER weka.classifiers.trees.J48 -U \n" + "    FILTER weka.filters.unsupervised.instance.Randomize \n" + "    DATASET iris.arff\n";
	}

	public static void main(String[] args) throws Exception
	{
		Weka5 wekanb;

		// parse command line

		// weka.classifiers.trees.J48

		String classifier = "weka.classifiers.bayes.NaiveBayes";
		String filter = "weka.filters.unsupervised.instance.Randomize";
		// String dataset = "dbworld_bodies.arff";

		// Using health dataset (dataset has been converted to .arff format)
		String dataset = "health.arff";
		Vector classifierOptions = new Vector();
		Vector filterOptions = new Vector();

		int i = 0;
		String current = "";
		boolean newPart = false;

		// run
		wekanb = new Weka5();
		wekanb.setClassifier(classifier, (String[]) classifierOptions.toArray(new String[classifierOptions.size()]));
		wekanb.setFilter(filter, (String[]) filterOptions.toArray(new String[filterOptions.size()]));
		wekanb.setTraining(dataset);
		wekanb.execute();
		System.out.println(wekanb.toString());
	}
}
